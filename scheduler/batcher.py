"""Continuous-batching scheduler: admits, prefills, decodes and evicts requests."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
import traceback
from typing import Optional

from ..engine.engine import Engine
from ..engine.request import Request


class ContinuousBatcher:
    def __init__(self, engine: Engine, max_prefills_per_step: int = 2):
        self.engine = engine
        self.max_prefills_per_step = max_prefills_per_step
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._lock = asyncio.Lock()
        self._wakeup = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        # Single worker: GPU work is serialised, but the event loop stays free.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def submit(self, prompt: str, max_tokens: int = 64) -> Request:
        req = self.engine.new_request(prompt, max_tokens)
        req.submitted_at = time.perf_counter()
        async with self._lock:
            self.waiting.append(req)
            self._wakeup.set()
        return req

    async def start(self):
        self.engine.set_event_loop(asyncio.get_running_loop())
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    async def _fail(self, r: Request):
        """Terminate a request that errored: free its slot, unblock consumer."""
        r.finished = True
        self.engine.release_slot(r.req_id)
        await r.token_queue.put(None)
        self.finished.append(r)

    async def _loop(self):
        loop = asyncio.get_running_loop()
        run = lambda fn, *a: loop.run_in_executor(self._executor, fn, *a)

        while True:
            try:
                # ── 1. Evict finished requests, release their slots ──
                still_running = []
                for r in self.running:
                    if r.finished:
                        self.engine.release_slot(r.req_id)
                        await r.token_queue.put(None)
                        self.finished.append(r)
                    else:
                        still_running.append(r)
                self.running = still_running

                # ── 2. Admit waiting requests (capped per iteration) ──
                async with self._lock:
                    n = min(len(self.waiting),
                            len(self.engine.free_slots),
                            self.max_prefills_per_step)
                    admit = self.waiting[:n]
                    del self.waiting[:n]

                for r in admit:
                    try:
                        ok = await run(self.engine.run_prefill, r)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[Batcher] prefill failed for req {r.req_id}: {e!r}")
                        await self._fail(r)
                        continue
                    if ok:
                        self.running.append(r)
                    else:  # no slot (shouldn't happen: we counted free slots)
                        async with self._lock:
                            self.waiting.insert(0, r)

                # ── 3. Idle → wait ───────────────────────────────────
                if not self.running and not self.waiting:
                    self._wakeup.clear()
                    await self._wakeup.wait()
                    continue

                # ── 4. One batched decode step ───────────────────────
                active = [r for r in self.running if not r.finished]
                if active:
                    try:
                        await run(self.engine.batch_decode, active)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[Batcher] decode failed, aborting batch: {e!r}")
                        for r in active:
                            r.finished = True  # evicted at the top of next iteration

                await asyncio.sleep(0)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                traceback.print_exc()
                print(f"[Batcher] scheduler error: {e!r}")
                await asyncio.sleep(0.1)

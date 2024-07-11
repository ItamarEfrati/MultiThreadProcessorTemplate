import logging
import queue
import threading
from abc import ABC, abstractmethod


class MicroService(ABC, threading.Thread):
    def __init__(self):
        super().__init__()
        self._input_queue = None
        self._output_queue = None
        self.is_running = True
        self.log = logging.getLogger()

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def file_name(self):
        pass

    @property
    def input_queue(self):
        return self._input_queue

    @property
    def output_queue(self):
        return self._output_queue

    @output_queue.setter
    def output_queue(self, queue):
        self._output_queue = queue

    @input_queue.setter
    def input_queue(self, queue):
        self._input_queue = queue

    def run(self):
        while self.is_running or not self.input_queue.empty():
            try:
                queue_values = self.input_queue.get(timeout=10)
                save_dir = queue_values[-1]
            except queue.Empty:
                continue
            try:
                output_queue_values = self.handle_queue_values(queue_values)
                self.save_results(output_queue_values, save_dir)
                output_queue_values = save_dir if output_queue_values is None else output_queue_values
                self.output_queue.put(output_queue_values)
            except Exception:
                self.log.exception(f'Fail handle Q values {queue_values}')
            finally:
                self.input_queue.task_done()

    @abstractmethod
    def handle_queue_values(self, queue_values):
        pass

    @abstractmethod
    def save_results(self, output_queue_values, save_dir):
        pass

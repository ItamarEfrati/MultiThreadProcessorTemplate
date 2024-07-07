import os
import time
import queue
import pathlib
import logging

from collections import defaultdict
from typing import List, Dict

from .constants import *
from .micro_services.micro_service import MicroService


class PipelineExecutor:

    def __init__(self,
                 services: Dict[str, List[MicroService]],
                 input_dir: str,
                 pipeline_type: str,
                 is_input_data_in_folder: bool
                 ):
        self.is_input_data_in_folder = is_input_data_in_folder
        self.input_directory = pathlib.Path(os.path.join(DATA_FOLDER, INPUT_FOLDER, input_dir))
        self.output_queues = defaultdict(lambda: queue.Queue())
        self.input_queues = defaultdict(lambda: queue.Queue())
        self.services = services
        folder_suffix = pipeline_type
        self.preprocess_folder = PREPROCESS_OUTCOMES + '_' + folder_suffix

        # if videos folder is none expecting the video source to be the laptop camera
        assert (
                self.input_directory.exists() and self.input_directory.is_dir()), f"check input folder {self.input_directory}"

        self.init_process_folder()
        self.log = logging.getLogger()

    def run(self):
        self.init_threads()
        self.wait_for_finish()

    # region Init
    def init_process_folder(self):
        pathlib.Path(DATA_FOLDER, PREPROCESS, self.preprocess_folder).mkdir(parents=True, exist_ok=True)

    def init_threads(self):
        first_q = queue.Queue()
        input_q = first_q
        for service_name, service_threads in self.services.items():
            self.input_queues[service_name] = input_q
            for single_thread in service_threads:
                single_thread.input_queue = input_q
                single_thread.output_queue = self.output_queues[service_name]
                single_thread.start()
            input_q = self.output_queues[service_name]

        self.feed_from_input(self.input_directory, first_q)

    def feed_from_input(self, input_folder, input_queue):
        self.log.info(f"Feeding inputs from {input_folder}")
        if any(input_folder.iterdir()):
            for directory in input_folder.iterdir():
                output_path = pathlib.Path(DATA_FOLDER, PREPROCESS, self.preprocess_folder, directory.name)
                output_path.mkdir(exist_ok=True, parents=True)
                q_input = tuple(directory.iterdir()) if self.is_input_data_in_folder else (directory.name,)
                q_input += (output_path,)

                input_queue.put(q_input)

    # endregion

    # region Threads

    def wait_for_finish(self):
        for service_name, service_threads in self.services.items():
            while self.input_queues[service_name].unfinished_tasks > len(service_threads):
                time.sleep(5)
            for single_thread in service_threads:
                single_thread.is_running = False
            for single_thread in service_threads:
                single_thread.join()
            self.log.info(f"{service_name} threads finished")
        self.log.info("All threads finish, shutting down program")

    # endregion

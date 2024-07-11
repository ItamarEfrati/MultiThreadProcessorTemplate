import os

from src.micro_services.micro_service import MicroService


class TextCounter(MicroService):

    @property
    def name(self):
        return 'text_counter'

    @property
    def file_name(self):
        return f'{self.name}_length.txt'

    def handle_queue_values(self, queue_values):
        file_path, save_dir = queue_values
        self.log.info(f"Measuring the size of {file_path}")
        return os.path.getsize(file_path)

    def save_results(self, output_queue_values, save_dir):
        file_size = output_queue_values
        with open(os.path.join(save_dir, self.file_name), 'w') as f:
            f.write(str(file_size))

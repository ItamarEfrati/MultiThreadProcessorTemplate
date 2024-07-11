import multiprocessing
from collections import defaultdict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import os

os.environ['NUMEXPR_MAX_THREADS'] = f'{multiprocessing.cpu_count()}'


def initiate_mini_services(service_config):
    services_dict = defaultdict(list)
    for micro_service in service_config.keys():
        if micro_service == 'pipeline_executor':
            continue
        micro_service_types = service_config.get(micro_service)
        services_name = list(filter(lambda x: not x.endswith('threads'), micro_service_types.keys()))
        for service_name in services_name:
            n_threads = micro_service_types.get(f"{service_name}_n_threads")
            services = []
            for i in range(n_threads):
                services.append(instantiate(micro_service_types.get(service_name)))
            services_dict[micro_service] += services
    return services_dict


@hydra.main(version_base='1.2', config_path=os.path.join('..', "config"), config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.preprocess:
        video_preprocess = instantiate(cfg.pipelines.pipeline_executor)
        micro_services = initiate_mini_services(cfg.pipelines)
        video_preprocess = video_preprocess(services=micro_services)
        video_preprocess.run()
    if cfg.train:
        dataloader = instantiate(cfg.dataloader)
        data = dataloader.load_data()
        study = instantiate(cfg.study)
        study.run_study(data)


if __name__ == '__main__':
    import sys

    sys.path.append(os.path.join(os.getcwd(), 'src'))
    main()

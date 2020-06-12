from src.configuration import cfg
from src.recording import Recording


class Subject:
    """
    This class defines a subject and stores relevant information for a subject (recordings, metadata,...)
    """

    def __init__(self, name, config=None):
        self.name = name
        self.recordings = {}
        if config is not None:
            if self.name in config["paths"]["subjects"]:
                base_path = f"{config['paths']['base']}{config['paths']['subjects'][self.name]['prefix']}"
                recording_paths = config["paths"]["subjects"][self.name]["recordings"]
                for data_type, recording_names in recording_paths.items():
                    recordings = [
                        Recording(
                            data_type,
                            f"{base_path}/offline/{name}-raw.pcl",
                            sampling_frequency=config["sampling_frequency"],
                            columns_to_remove=config["columns_to_remove"],
                            default_crop_secs=10,
                        )
                        for name in recording_names
                    ]
                    self.recordings[data_type] = recordings


class SubjectFactory:
    """
    Helper factory used to get an instance of a subject more easily
    """

    @staticmethod
    def get_subject(name):
        return Subject(name, config=cfg)


if __name__ == "__main__":
    subject = SubjectFactory.get_subject("adelie")
    rec1 = subject.recordings["baseline"][0]
    rec2 = subject.recordings["meditation"][1]
    rec3 = rec1 + rec2
    print(rec3.bandpower_by_epoch())

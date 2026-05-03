from registrable import Registrable


class Runner(Registrable):
    @staticmethod
    def build_from_config(config):
        runner_cls = Runner.by_name(config["runner_name"].lower())
        return runner_cls(config)

    def run(self):
        """Base run method"""
        pass

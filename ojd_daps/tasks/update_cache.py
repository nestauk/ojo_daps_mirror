from daps_utils import DapsRootTask, MetaflowTask

from ojd_daps import config as CONFIG

config = CONFIG["update_cache"]


class RootTask(DapsRootTask):
    def requires(self):
        tag = "production" if self.production else "dev"
        yield MetaflowTask(
            flow_path="aggregate/update_cache.py",
            flow_tag=tag,
            force=True,  # Always force rerun
            **config[tag],
        )

from pathlib import Path
import importlib.util as importer


def test_plot_modules():
    pwd = Path(__file__).parent.parent / "plots"
    paths = (path for path in pwd.iterdir() if path.suffix == ".py")
    for path in paths:
        # Load the module
        spec = importer.spec_from_file_location("DUMMY", path)
        module = importer.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Check that the module has the correct functions
        getattr(module, "make_plot_data")
        getattr(module, "make_plot")

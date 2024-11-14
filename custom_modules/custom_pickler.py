import dill as pickle
import hashlib
import os


class DillWrapper:
    """
    A class to wrap a function and provide methods to pickle and unpickle its output.
    Attributes:
    -----------
        function : callable
            The function to be wrapped.
        kwargs : dict
            Additional keyword arguments to be passed to the function.
        full_path : str
            The absolute path of the current working directory.
        dill_verbose : bool
            Flag to enable verbose output.
    Methods:
    --------
        pickle_object(obj, file_name: str, overwrite: bool=False):
            Pickles the given object to a file.
        dump(overwrite: bool=False):
            Executes the wrapped function, pickles its output, and returns the output.
        load(create_if_not_exist: bool=False):
            Loads the pickled object from a file. If the file does not exist, optionally creates it.
        hash_object():
            Generates an MD5 hash based on the function name and its keyword arguments.
        __call__():
            Calls the wrapped function with the provided keyword arguments.
    """

    def __init__(self, function: callable = None, dill_verbose: bool = False, **kwargs):
        self.function = function
        self.kwargs = kwargs
        self.full_path = os.path.abspath(os.getcwd())
        self.verbose = dill_verbose

    def pickle_object(self, obj, file_name: str, overwrite: bool = False):
        path = self.full_path + "/PickleJar/"
        file_name = path + file_name

        if not file_name.endswith(".pkl"):
            file_name += ".pkl"
        if os.path.isfile(file_name):
            if not overwrite:
                if self.verbose:
                    print(f"File {file_name} already exists. Not overwriting...")
                return obj
            else:
                if self.verbose:
                    print(f"File {file_name} already exists. Overwriting...")
                with open(file_name, "wb") as file:
                    pickle.dump(obj, file)
                return obj
        else:
            if self.verbose:
                print(f"File {file_name} does not exist. Creating...")
            with open(file_name, "wb") as file:
                pickle.dump(obj, file)
            return obj

    def dump(self, overwrite: bool = False):
        p_obj = self.function(**self.kwargs)
        return self.pickle_object(
            obj=p_obj, file_name=self.hash_object(), overwrite=overwrite
        )

    def load(self, create_if_not_exist: bool = False):
        path = self.full_path + "/PickleJar/"
        file_name = path + self.hash_object() + ".pkl"

        if os.path.isfile(file_name):
            with open(file_name, "rb") as file:
                return pickle.load(file)
        else:
            if create_if_not_exist:
                return self.dump()
            else:
                raise FileNotFoundError(f"File {file_name} does not exist.")

    def load_by_hash(self, hash_str: str):
        path = self.full_path + "/PickleJar/"
        file_name = path + hash_str + ".pkl"

        if os.path.isfile(file_name):
            with open(file_name, "rb") as file:
                return pickle.load(file)
        else:
            raise FileNotFoundError(f"File {file_name} does not exist.")

    def remove(self):
        path = self.full_path + "/PickleJar/"
        file_name = path + self.hash_object() + ".pkl"
        if os.path.isfile(file_name):
            os.remove(file_name)
            if self.verbose:
                print(f"File {file_name} removed.")
            return True
        else:
            if self.verbose:
                print(f"File {file_name} does not exist.")
            return False

    def hash_object(self):
        text = str(self.function.__name__) + str(self.kwargs)
        hash_obj = hashlib.md5(text.encode())
        return hash_obj.hexdigest()

    def __call__(self):
        return self.function(**self.kwargs)

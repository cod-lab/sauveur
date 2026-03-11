from functools import wraps
from typing import get_type_hints
from pprint import pprint as pp


def validate_args(fnc_model):
    def decorator(fnc):
        @wraps(fnc)
        def wrapper(self, *args, **kwargs):
            all_vars_with_types = get_type_hints(fnc)     # get all param names with their type
            all_vars_names = [*all_vars_with_types]

            args_params={}
            for i in range(len(args)):
                args_params[all_vars_names[i]] = args[i]

            all_params = args_params|kwargs         # storing args & kwargs in one dict

            model_obj = fnc_model(**all_params)     # validating fnc params thru pydantic model
            return fnc(self, **model_obj.model_dump())     # passing all params and calling fnc
        return wrapper
    return decorator



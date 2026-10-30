"""G10 namespace on the original persistent serial SR supervisor."""
from bridge import serial

if __name__ == '__main__':
    module = serial()
    module.__dict__['__name__'] = '__main__'
    exec(compile(module.original_entry, module.__file__, 'exec'), module.__dict__)

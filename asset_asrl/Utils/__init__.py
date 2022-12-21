from asset.Utils import *
import asset as _asset
import inspect

      
get_core_count = _asset.Utils.get_core_count


if __name__ == "__main__":
    mlist = inspect.getmembers(_asset.Utils)
    for m in mlist:print(m[0],'= _asset.Utils.'+str(m[0]))
  
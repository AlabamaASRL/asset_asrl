import asset as _asset
import asset_asrl.VectorFunctions
import asset_asrl.OptimalControl
import asset_asrl.Utils
import asset_asrl.Astro
import asset_asrl.Solvers
import inspect

SoftwareInfo = _asset.SoftwareInfo

if __name__ == "__main__":
    _asset.SoftwareInfo()
    mlist = inspect.getmembers(_asset)
    for m in mlist:print(m[0],'= _asset.'+str(m[0]))
    
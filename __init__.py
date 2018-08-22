import json
wlenet_path = __path__[0]
with open(wlenet_path + '/config.json') as json_fid:
    config = json.load(json_fid)

from astropy import log
log.setLevel('WARNING')

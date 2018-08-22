import numpy as np


def crop_center(img, out_szx, out_szy):
    
    inp = img
    out = np.zeros((out_szy, out_szx))
    inp_szy, inp_szx = inp.shape    
    
    inp_x0 = np.maximum(inp_szx//2 - out_szx//2, 0)
    inp_x1 = np.minimum(inp_szx//2 + out_szx//2, inp_szx)
    out_xm = (inp_x1 - inp_x0) // 2
    out_xp = inp_x1 - inp_x0 - out_xm
    out_x0 = out_szx//2 - out_xm
    out_x1 = out_szx//2 + out_xp

    inp_y0 = np.maximum(inp_szy//2 - (out_szy//2), 0)
    inp_y1 = np.minimum(inp_szy//2 + (out_szy//2), inp_szy)
    out_ym = (inp_y1 - inp_y0) // 2
    out_yp = inp_y1 - inp_y0 - out_ym
    out_y0 = out_szy//2 - out_ym
    out_y1 = out_szy//2 + out_yp
    
    out[out_y0:out_y1, out_x0:out_x1] = inp[inp_y0:inp_y1, inp_x0:inp_x1]    
    return out
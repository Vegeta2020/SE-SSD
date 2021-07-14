#from .mg_head import Head, MultiGroupHead, RegHead

# mg_head: default version
# mg_head_ciassd: !!! Faster version of CIA-SSD !!!
# mg_head_sessd: work with mean teacher framework. !!! Work and Fixed !!! The cls_thres may be modified.


from .mg_head_sessd import MultiGroupHead

__all__ = ["MultiGroupHead"]

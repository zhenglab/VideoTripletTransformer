import torch
import torch.nn.functional as f


#def evaluation(name, fake, real, mask):
#    b,c,w,h = real.size()
#    mse_score = f.mse_loss(fake, real)
#    fore_area = torch.sum(mask)
#    fmse_score = f.mse_loss(fake*mask,real*mask)*w*h/fore_area
#    mse_score = mse_score.item()
#    fmse_score = fmse_score.item()
#    # score_str = "%s MSE %0.2f | fMSE %0.2f" % (name, mse_score,fmse_score)
#    image_fmse_info = (name, round(fmse_score,2), round(mse_score, 2))
#    return mse_score, fmse_score, image_fmse_info
    
# with fmse between comp and real
def evaluation(name, fake, real, comp, mask):
    b,c,w,h = real.size()
    mse_score = f.mse_loss(fake, real)
    fore_area = torch.sum(mask)
    fmse_score = f.mse_loss(fake*mask,real*mask)*w*h/fore_area
    # add code
    comp_mse_score = f.mse_loss(comp, real)
    comp_fmse_score = f.mse_loss(comp*mask,real*mask)*w*h/fore_area
    # -------------
    mse_score = mse_score.item()
    fmse_score = fmse_score.item()
    # add code
    comp_mse_score = comp_mse_score.item()
    comp_fmse_score = comp_fmse_score.item()
    # -------------
    # score_str = "%s MSE %0.2f | fMSE %0.2f" % (name, mse_score,fmse_score)
    # image_fmse_info = (name, round(fmse_score,2), round(mse_score, 2))
    image_fmse_info = (name, round(fmse_score,2), round(mse_score, 2), round(comp_fmse_score, 2))
    return mse_score, fmse_score, comp_mse_score, comp_fmse_score, image_fmse_info
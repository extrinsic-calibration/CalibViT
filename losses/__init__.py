try:
	from .chamfer_loss import ChamferDistanceLoss
except:
	print("Sorry ChamferDistance loss is not compatible with your system!")


from .photo_loss import PhotoLoss
from .mse_loss import MSETransformationLoss
from .quaternion_loss import QuaternionLoss
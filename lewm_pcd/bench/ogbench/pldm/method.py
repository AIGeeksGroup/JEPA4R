"""Current RGB Impala512 PLDM model, objective and optimizer factory."""
import importlib
from pathlib import Path
import sys

AUTHOR_COMMIT = "1bd7e564ecd961205bc18b23067b19e9ca24ac90"
OBSERVATIONS, ACTIONS, ACTION_DIM, IMAGE_SIZE, LATENT_DIM = 16, 15, 5, 65, 512


def check_batch(pixels, actions):
    batch = pixels.shape[0]
    if tuple(pixels.shape) != (batch, 16, 3, 65, 65) or batch < 1:
        raise ValueError("pixels must be [B, 16, 3, 65, 65], full-field RGB")
    if tuple(actions.shape) != (batch, 15, 5):
        raise ValueError("normalized actions must be [B, 15, 5], without chunks")


def _official(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    package = importlib.import_module("pldm")
    if Path(package.__file__).resolve() != source / "pldm/__init__.py":
        raise ValueError("another pldm package was already imported")


def build_model(official_source, *, pretrained=False):
    if pretrained is not False:
        raise ValueError("PLDM must start fresh, without pretrained weights")
    _official(official_source)
    import torch
    from pldm.models.jepa import JEPA, JEPAConfig
    from pldm.models.encoders.enums import BackboneConfig
    from pldm.models.encoders.impala import ImpalaEncoder
    from pldm.models.enums import PredictorConfig
    from pldm.models.predictors import build_predictor
    from pldm.models.misc import Projector, MLP
    from pldm.objectives.vicreg import VICRegObjective, VICRegObjectiveConfig
    from pldm.objectives.idm import IDMObjective, IDMObjectiveConfig

    class RGBJEPA(JEPA):
        def __init__(self):
            # Reuse native forward; replace only its non-forwarded RGB factory.
            torch.nn.Module.__init__(self)
            self.config = JEPAConfig(
                backbone=BackboneConfig(arch="impala", channels=3, final_ln=True),
                predictor=PredictorConfig(predictor_arch="rnnV2",
                    predictor_subclass="512-512", rnn_layers=1, z_dim=0,
                    residual=True, predictor_ln=True, tie_backbone_ln=True),
                action_dim=5, momentum=0)
            self.use_propio_pos = self.use_propio_vel = False
            self.l2 = False
            self.backbone = ImpalaEncoder(input_channels=3, final_ln=True)
            self.backbone.config = self.config.backbone
            self.backbone.input_dim = (3, 65, 65)
            self.backbone.output_dim = self.backbone.output_obs_dim = 512
            self.backbone.output_propio_dim = 0
            self.spatial_repr_dim = self.repr_dim = 512
            self.backbone_ema = None
            self.predictor = build_predictor(self.config.predictor,
                repr_dim=512, action_dim=5, pred_propio_dim=0,
                pred_obs_dim=512, backbone_ln=self.backbone.final_ln)

    class DeviceVICReg(VICRegObjective):
        def __init__(self):
            # Upstream constructor forces cuda; inherited loss methods unchanged.
            torch.nn.Module.__init__(self)
            self.config = VICRegObjectiveConfig(projector="id", sim_coeff=1.0,
                std_coeff=3.9843, cov_coeff=6.9238, std_coeff_t=0.24535,
                cov_coeff_t=0.0, sim_coeff_t=0.74242, adjust_cov=True)
            self.name_prefix, self.pred_attr = "l1", "state"
            self.projector = Projector(arch="id", embedding=512, random=False)

    class DeviceIDM(IDMObjective):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = IDMObjectiveConfig(coeff=1.072, action_dim=5,
                arch="512", arch_subclass="a", use_pred=False)
            self.name_prefix = "l1"
            self.action_predictor = MLP(arch="512", input_dim=1024, output_shape=5)

    class ImagePLDM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core, self.vicreg, self.idm = RGBJEPA(), DeviceVICReg(), DeviceIDM()
            self.embed_dim = 512

        def encode(self, images):
            if tuple(images.shape[1:]) != (3, 65, 65):
                raise ValueError("encode requires preprocessed RGB [B,3,65,65]")
            return self.core.backbone(images).encodings

        def forward_result(self, pixels, actions):
            check_batch(pixels, actions)
            if not pixels.is_floating_point() or not actions.is_floating_point():
                raise ValueError("pixels and normalized actions must be floating tensors")
            return self.core.forward_posterior(pixels.transpose(0, 1), actions.transpose(0, 1))

        def forward(self, pixels, actions):
            result = self.forward_result(pixels, actions)
            vic = self.vicreg(None, [result])
            idm = self.idm(None, [result])
            return {"loss": vic.total_loss + idm.total_loss,
                "prediction_loss": vic.sim_loss, "std_loss": vic.std_loss,
                "cov_loss": vic.cov_loss, "std_time_loss": vic.std_loss_t,
                "cov_time_loss": vic.cov_loss_t, "temporal_loss": vic.sim_loss_t,
                "idm_loss": idm.action_loss,
                "embedding": result.backbone_output.encodings.transpose(0, 1)}

    return ImagePLDM()


def build_optimizer(model, *, base_lr=0.0007):
    """Original Adam/wd; all parameters include the explicitly corrected IDM."""
    import torch
    return torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-6)


def build_scheduler(optimizer, *, epochs, batch_steps, batch_size=128, base_lr=0.0007):
    """Native schedule: 10% warmup, B/256 peak scaling, cosine to 0.001 peak."""
    from pldm.optimizers.schedulers import Scheduler, LRSchedule
    return Scheduler(schedule=LRSchedule.Cosine, base_lr=base_lr, data_loader=None,
        epochs=epochs, optimizer=optimizer, batch_steps=batch_steps, batch_size=batch_size)

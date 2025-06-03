from ..base import Base


class Resampler(Base):
    """Resampler base class"""
    def update_resample_index(self,new_particles, index):
        if hasattr(new_particles,'resample_index'):
            index=index.astype(int)
            new_particles.resample_index=index
            # if resampling with a model with a time-varying driver, need to resample the mu values as well:
            model=new_particles.hypothesis.prediction.transition_model
            #TODO: change code to deal with combined transition model more reliably.
            if isinstance(model.mu_W_state, list):
                for sub_model in model.model_list:
                    self.resample_mu_driver(sub_model,index)
            else:
                self.resample_mu_driver(model,index)

import logging
import copy

import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
logger = logging.getLogger()
from .ptqnet import PQTANet
from .rnet import RNet
from .loss import CrossEntropyLossForChecklist
from config.args import override_model_args

class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    def __init__(self, args,state_dict=None):
        # Book-keeping.
        self.args = args
        self.updates = 0
        self.use_cuda = False
        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type.lower() == 'pqtanet':
            self.network = PQTANet(args)
        elif args.model_type.lower() == 'rnet':
            self.network = RNet(args)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            self.network.load_state_dict(state_dict)
    def init_loss(self,args):
        self.criterion = CrossEntropyLossForChecklist()
    def init_lr_scheduler(self,args,num_training_steps):
        self.lr_scheduler = LinearDecayWithWarmup(
            args.learning_rate, num_training_steps, args.warmup_proportion)
    def init_optimizer(self,args):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: network parameters
        """
        if args.optimizer == 'sgd':
            self.optimizer = paddle.optimizer.SGD(
                learning_rate=self.lr_scheduler,
                parameters=self.network.parameters(),
                weight_decay=args.weight_decay
            )
        elif args.optimizer == "momentum":
            self.optimizer = paddle.optimizer.Momentum(
                learning_rate=self.lr_scheduler,
                momentum=args.momentum_param,
                parameters=self.network.parameters(),
                use_nesterov=True,
                weight_decay=args.weight_decay)
        elif args.optimizer == 'adamax':
            self.optimizer = paddle.optimizer.Adamax(
                learning_rate=self.lr_scheduler,
                epsilon=args.adamax_epsilon,
                parameters=self.network.parameters(),
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'adadelta':
            self.optimizer = paddle.optimizer.Adadelta(
                learning_rate=self.lr_scheduler,
                epsilon=args.adadelta_epsilon,
                parameters=self.network.parameters(),
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'adam':
            self.optimizer = paddle.optimizer.Adam(
                    learning_rate=self.lr_scheduler,
                    epsilon=args.adam_epsilon,
                    parameters=self.network.parameters(),
                    weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            self.optimizer = paddle.optimizer.AdamW(
                    learning_rate=self.lr_scheduler,
                    epsilon=args.adam_epsilon,
                    parameters=self.network.parameters(),
                    weight_decay=args.weight_decay,
                    apply_decay_param_fun=lambda x: x in [
                            p.name for n, p in self.network.named_parameters()
                            if not any(nd in n for nd in ["bias", "norm"])
                    ])
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self,batch):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.network.train()
        # Transfer to GPU
        # ex : x,x_c,x_f,x_mask,targets
        docs_c,docs_w,tite_c,tite_w,ques_c,ques_w,y_s,y_e = batch
        input_ids, segment_ids, start_positions, end_positions, answerable_label = batch
        # Run forward
        logits = self.network(input_ids=input_ids, token_type_ids=segment_ids)

        # Compute loss and accuracies
        loss = self.criterion(logits, (start_positions, end_positions,answerable_label))
        np_loss = loss.numpy()
        # Clear gradients and run backward
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_gradients()

        return np_loss
    def predict(self,batch):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
        Output:
            scores: batch * fine_second_size * class_number predict matrix
            targets: batch * fine_second_size  labeled data
        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        input_ids, segment_ids = batch
        start_logits_tensor, end_logits_tensor, cls_logits_tensor = self.network(input_ids, segment_ids)
        # Decode predictions
        return start_logits_tensor, end_logits_tensor, cls_logits_tensor
    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------
    def save(self, filename):
        model_to_save = self.network._layers if isinstance(
            self.network, paddle.DataParallel) else self.network
        state_dict = copy.copy(model_to_save.state_dict())
        params = {
            'state_dict': state_dict,
            'args': self.args,
        }
        try:
            paddle.save(params,filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def save_checkpoint(self, filename,epoch):
        model_to_save = self.network._layers if isinstance(
            self.network, paddle.DataParallel) else self.network
        params = {
            'state_dict': model_to_save.state_dict(),
            'args': self.args,
            'epoch': epoch
        }
        try:
            paddle.save(params,filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = paddle.load(filename)
        args = saved_params['args']
        state_dict = saved_params['state_dict']
        if new_args:
            args = override_model_args(args, new_args)
        return DocReader(args,state_dict)
    @staticmethod
    def load_checkpoint(filename):
        logger.info('Loading model %s' % filename)
        saved_params = paddle.load(filename)
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        args = saved_params['args']
        model = DocReader(args,state_dict)
        model.init_optimizer(args)
        return model, epoch

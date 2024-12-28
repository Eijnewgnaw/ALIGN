from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import instance_contrastive_Loss, category_contrastive_loss
from utils import classify
from utils.next_batch import next_batch_gt, next_batch, next_batch_3view, next_batch_gt_3view


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.LayerNorm(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent





class ALIGN():
    def __init__(self, config):
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        self._config = config

        self._latent_dim = config['Autoencoder']['arch1'][-1]

        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder3 = Autoencoder(config['Autoencoder']['arch3'], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm'])



        # Define generators for cross-view latent representation mapping
        self.a2b_gen = Generator(self._latent_dim, self._latent_dim)
        self.b2a_gen = Generator(self._latent_dim, self._latent_dim)
        self.a2c_gen = Generator(self._latent_dim, self._latent_dim)
        self.c2a_gen = Generator(self._latent_dim, self._latent_dim)
        self.b2c_gen = Generator(self._latent_dim, self._latent_dim)
        self.c2b_gen = Generator(self._latent_dim, self._latent_dim)

        # Define discriminators for adversarial learning
        self.a2b_disc = Discriminator(self._latent_dim)
        self.b2a_disc = Discriminator(self._latent_dim)
        self.a2c_disc = Discriminator(self._latent_dim)
        self.c2a_disc = Discriminator(self._latent_dim)
        self.b2c_disc = Discriminator(self._latent_dim)
        self.c2b_disc = Discriminator(self._latent_dim)


    def train_completegraph_supervised(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train, x1_test,
                                       x2_test, x3_test, labels_train, labels_test, mask_train, mask_test, optimizer,
                                       device):
        """Training the model with complete graph for classification"""
        epochs = config['training']['epoch']
        batch_size = config['training']['batch_size']

        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask_train == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)

        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_view3 = x3_train[flag]

        GT = torch.from_numpy(labels_train).long().to(device)[flag]
        classes = np.unique(np.concatenate([labels_train, labels_test])).size
        flag_gt = False
        if torch.min(GT) == 1:
            flag_gt = True

        for k in range(epochs):
            X1, X2, X3, gt = shuffle(train_view1, train_view2, train_view3, GT)
            all_ccl, all0, all1, all2, all_icl, all_gen, all_disc = 0, 0, 0, 0, 0, 0, 0

            for batch_x1, batch_x2, batch_x3, gt_batch, batch_No in next_batch_gt_3view(X1, X2, X3, gt, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)
                reconstruction_loss = recon1 + recon2 + recon3

                # Instance-level contrastive loss
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl3 = instance_contrastive_Loss(z_half2, z_half3, config['training']['alpha'])
                loss_icl = (loss_icl1 + 0.1 * loss_icl2 + 0.1 * loss_icl3) / 3

                # GAN-based loss
                a2b = self.a2b_gen(z_half1)
                b2a = self.b2a_gen(z_half2)
                a2c = self.a2c_gen(z_half1)
                c2a = self.c2a_gen(z_half3)
                b2c = self.b2c_gen(z_half2)
                c2b = self.c2b_gen(z_half3)

                # Generator loss
                gen_loss_a2b = F.mse_loss(a2b, z_half2) - torch.mean(self.a2b_disc(a2b))
                gen_loss_b2a = F.mse_loss(b2a, z_half1) - torch.mean(self.b2a_disc(b2a))
                gen_loss_a2c = F.mse_loss(a2c, z_half3) - torch.mean(self.a2c_disc(a2c))
                gen_loss_c2a = F.mse_loss(c2a, z_half1) - torch.mean(self.c2a_disc(c2a))
                gen_loss_b2c = F.mse_loss(b2c, z_half3) - torch.mean(self.b2c_disc(b2c))
                gen_loss_c2b = F.mse_loss(c2b, z_half2) - torch.mean(self.c2b_disc(c2b))

                gen_loss = (gen_loss_a2b + gen_loss_b2a + gen_loss_a2c +
                            gen_loss_c2a + gen_loss_b2c + gen_loss_c2b)

                # Discriminator loss
                disc_loss_a2b = -torch.mean(self.a2b_disc(z_half2)) + torch.mean(self.a2b_disc(a2b.detach()))
                disc_loss_b2a = -torch.mean(self.b2a_disc(z_half1)) + torch.mean(self.b2a_disc(b2a.detach()))
                disc_loss_a2c = -torch.mean(self.a2c_disc(z_half3)) + torch.mean(self.a2c_disc(a2c.detach()))
                disc_loss_c2a = -torch.mean(self.c2a_disc(z_half1)) + torch.mean(self.c2a_disc(c2a.detach()))
                disc_loss_b2c = -torch.mean(self.b2c_disc(z_half3)) + torch.mean(self.b2c_disc(b2c.detach()))
                disc_loss_c2b = -torch.mean(self.c2b_disc(z_half2)) + torch.mean(self.c2b_disc(c2b.detach()))

                disc_loss = (disc_loss_a2b + disc_loss_b2a + disc_loss_a2c +
                             disc_loss_c2a + disc_loss_b2c + disc_loss_c2b)

                # Category-level contrastive loss
                loss_ccl = category_contrastive_loss(torch.cat([z_half1, z_half2, z_half3], dim=1), gt_batch, classes,
                                                     flag_gt)

                # Total loss
                all_loss = (
                        loss_icl +
                        reconstruction_loss * config['training']['lambda2'] +
                        gen_loss * config['training']['lambda3'] +
                        disc_loss * config['training']['lambda4'] +
                        loss_ccl
                )

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all_icl += loss_icl.item()
                all_ccl += loss_ccl.item()
                all_gen += gen_loss.item()
                all_disc += disc_loss.item()
                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()

            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Loss_icl = {:.4e} ===> Los_ccl = {:.4e} ===> Generator loss = {:.4e} " \
                     "===> Discriminator loss = {:.4e} ===> All loss = {:.4e}" \
                .format((k + 1), epochs, all1, all2, all_icl, all_ccl, all_gen, all_disc, all0)

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                # if True:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
                    self.a2b_gen.eval(), self.b2a_gen.eval()
                    self.b2c_gen.eval(), self.c2b_gen.eval()
                    self.a2c_gen.eval(), self.c2a_gen.eval()

                    # Training data
                    a_idx_eval = mask_train[:, 0] == 1
                    b_idx_eval = mask_train[:, 1] == 1
                    c_idx_eval = mask_train[:, 2] == 1
                    a_missing_idx_eval = mask_train[:, 0] == 0
                    b_missing_idx_eval = mask_train[:, 1] == 0
                    c_missing_idx_eval = mask_train[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_train[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_train[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_train[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
                        ano_bonlyhas= self.b2a_gen(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
                        ano_conlyhas = self.c2a_gen(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a_gen(ano_bcbothhas_1)[0] + self.c2a_gen(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
                        bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

                        bno_aonlyhas = self.autoencoder1.encoder(x1_train[bno_aonlyhas_idx])
                        bno_aonlyhas= self.a2b_gen(bno_aonlyhas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_train[bno_conlyhas_idx])
                        bno_conlyhas = self.c2b_gen(bno_conlyhas)

                        bno_acbothhas_1 = self.autoencoder1.encoder(x1_train[bno_acbothhas_idx])
                        bno_acbothhas_2 = self.autoencoder3.encoder(x3_train[bno_acbothhas_idx])
                        bno_acbothhas = (self.a2b_gen(bno_acbothhas_1)[0] + self.c2b_gen(bno_acbothhas_2)[0]) / 2.0

                        latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
                        latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

                    if c_missing_idx_eval.sum() != 0:
                        #   b缺
                        cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
                        cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_train[cno_aonlyhas_idx])
                        cno_aonlyhas = self.a2c_gen(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_train[cno_bonlyhas_idx])
                        cno_bonlyhas = self.b2c_gen(cno_bonlyhas)

                        cno_abbothhas_1 = self.autoencoder1.encoder(x1_train[cno_abbothhas_idx])
                        cno_abbothhas_2 = self.autoencoder2.encoder(x2_train[cno_abbothhas_idx])
                        cno_abbothhas = (self.a2c_gen(cno_abbothhas_1)[0] + self.b2c_gen(cno_abbothhas_2)[0]) / 2.0

                        latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
                        latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                                    dim=1).cpu().numpy()

                    # Test data
                    a_idx_eval = mask_test[:, 0] == 1
                    b_idx_eval = mask_test[:, 1] == 1
                    c_idx_eval = mask_test[:, 2] == 1
                    a_missing_idx_eval = mask_test[:, 0] == 0
                    b_missing_idx_eval = mask_test[:, 1] == 0
                    c_missing_idx_eval = mask_test[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_test[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_test[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_test[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_test.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_test.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_test.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_test[ano_bonlyhas_idx])
                        ano_bonlyhas= self.b2a_gen(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_test[ano_conlyhas_idx])
                        ano_conlyhas= self.c2a_gen(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_test[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_test[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a_gen(ano_bcbothhas_1)[0] + self.c2a_gen(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
                        bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

                        bno_aonlyhas = self.autoencoder1.encoder(x1_test[bno_aonlyhas_idx])
                        bno_aonlyhas = self.a2b_gen(bno_aonlyhas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_test[bno_conlyhas_idx])
                        bno_conlyhas = self.c2b_gen(bno_conlyhas)

                        bno_acbothhas_1 = self.autoencoder1.encoder(x1_test[bno_acbothhas_idx])
                        bno_acbothhas_2 = self.autoencoder3.encoder(x3_test[bno_acbothhas_idx])
                        bno_acbothhas = (self.a2b_gen(bno_acbothhas_1)[0] + self.c2b_gen(bno_acbothhas_2)[0]) / 2.0

                        latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
                        latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

                    if c_missing_idx_eval.sum() != 0:
                        cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
                        cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_test[cno_aonlyhas_idx])
                        cno_aonlyhas = self.a2c_gen(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_test[cno_bonlyhas_idx])
                        cno_bonlyhas = self.b2c_gen(cno_bonlyhas)

                        cno_abbothhas_1 = self.autoencoder1.encoder(x1_test[cno_abbothhas_idx])
                        cno_abbothhas_2 = self.autoencoder2.encoder(x2_test[cno_abbothhas_idx])
                        cno_abbothhas = (self.a2c_gen(cno_abbothhas_1)[0] + self.b2c_gen(cno_abbothhas_2)[0]) / 2.0

                        latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
                        latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion_test = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                                   dim=1).cpu().numpy()

                    from sklearn.metrics import accuracy_score
                    from sklearn.metrics import precision_score
                    from sklearn.metrics import f1_score

                    label_pre = classify.ave(latent_fusion_train, latent_fusion_test, labels_train)

                    scores = accuracy_score(labels_test, label_pre)

                    precision = precision_score(labels_test, label_pre, average='macro')
                    precision = np.round(precision, 2)

                    f_score = f1_score(labels_test, label_pre, average='macro')
                    f_score = np.round(f_score, 2)

                    accumulated_metrics['acc'].append(scores)
                    accumulated_metrics['precision'].append(precision)
                    accumulated_metrics['f_measure'].append(f_score)
                    logger.info('\033[2;29m Accuracy on the test set is {:.4f}'.format(scores))
                    logger.info('\033[2;29m Precision on the test set is {:.4f}'.format(precision))
                    logger.info('\033[2;29m F_score on the test set is {:.4f}'.format(f_score))

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
                    self.a2b_gen.train(), self.b2a_gen.train()
                    self.b2c_gen.train(), self.c2b_gen.train()
                    self.a2c_gen.train(), self.c2a_gen.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['precision'][-1], accumulated_metrics['f_measure'][
            -1]
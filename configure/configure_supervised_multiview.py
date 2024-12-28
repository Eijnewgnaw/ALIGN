def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            type='CG',  # other: CV
            view=3,
            seed=8,
            training=dict(
                lr=3.0e-4,
                start_dual_prediction=500,
                batch_size=256,
                epoch=1000,
                alpha=10,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=1.0,  # Generator Loss 权重
                lambda4=1.0,  # Discriminator Loss 权重
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                arch3=[928, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            type='CG',
            view=3,
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                arch3=[40, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            training=dict(
                lr=1.0e-3,
                start_dual_prediction=0,
                batch_size=512,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1,
                lambda3 = 1.0,  # Generator Loss 权重
                lambda4 = 1.0,  # Discriminator Loss 权重
            ),
            seed=8,
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            type='CG',  # other: CV
            view=3,
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 40],
                arch2=[40, 1024, 1024, 1024, 40],
                arch3=[20, 1024, 1024, 1024, 40],
                activations='relu',
                batchnorm=True,
            ),
            training=dict(
                lr=1.0e-4,
                start_dual_prediction=0,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1,
                lambda3=1.0,  # Generator Loss 权重
                lambda4=1.0,  # Discriminator Loss 权重
            ),
            seed=2,
        )
    elif data_name in ['20newsgroups']:
        """The default configs."""
        return dict(
            view=3,
            seed=2,
            type='CG',
            training=dict(
                lr=1.0e-4,
                batch_size=64,
                epoch=500,
                start_dual_prediction=0,
                lambda2=0.1,
                lambda1=0.1,
                lambda3=1.0,  # Generator Loss 权重
                lambda4=1.0,  # Discriminator Loss 权重
                alpha=10,
            ),
            Autoencoder=dict(
                arch1=[2000, 1024, 1024, 1024, 1024],
                arch2=[2000, 1024, 1024, 1024, 1024],
                arch3=[2000, 1024, 1024, 1024, 1024],
                activations='relu',
                batchnorm=True,
            ),
        )

    else:
        raise Exception('Undefined data name')

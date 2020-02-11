from CSA import CSAModel


def create_model(opt):

    print(opt.model)
    if opt.model == 'csa_net':
        model = CSAModel()
    else:
        raise ValueError("Model [%s] con't be recognized." % opt.model)

    model.initialize(opt)
    print("Model [%s] is created." % (model.name()))

    return model

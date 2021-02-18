# 以下代码用于修改不同层之间的网络参数并单独设定学习率，以及基于测量值修改学习率代码
# model是实例化后的网络结构
ignored_params1 = list(map(id, model.module.out_module.parameters()))
ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
ignored_params3 = list(map(id, model.module.Convmask.parameters()))
ignored_params = ignored_params1 + ignored_params2 + ignored_params3

base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

opt = torch.optim.SGD([{"params": base_params},
                       {"params": model.module.out_module.parameters(), "lr": 5 * 0.01},
                       {"params": model.module.classifier_swap.parameters(), "lr": 5 * 0.01},
                       {"params": model.module.Convmask.parameters(), "lr": 5 * 0.01}], lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode="max", factor=0.5, patience=10,
                                                       cooldown=2, min_lr=0.000001, verbose=True)
scheduler.step((acc / len(test_datasets)) * 100)


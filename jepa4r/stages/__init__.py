"""Pipeline stages. Each module exposes ``run(cfg, out_dir) -> dict | None``; a returned dict is written to
``out_dir/metrics.json`` together with the run key (stage, env, method, decoder, tag).

Run one with ``jepa4r-run <stage> key=value ...``: the CLI composes the config, creates ``out_dir``
(jepa4r.config.run_dir), saves ``config.yaml`` there, calls ``run`` and finally writes ``DONE``.

    collect_wm_data ----------> train_wm ----------> train_policy      Tab. 1, Tab. 2, Tab. 4b-d
    prepare_policy_data ---------------------------------^
    train_wm (cube_single, jepa4r, epoch 10) --> train_decoder --> eval_recon   Tab. 3, Tab. 4a
    verify_twin (rm75b_cube); twin-trained train_wm / train_policy / train_decoder --> eval_preview   Sec. 5.6
    count_params                                                        Sec. 4.2 / 4.4 parameter counts
"""

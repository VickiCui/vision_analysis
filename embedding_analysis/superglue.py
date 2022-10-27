import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import jiant.scripts.download_data.runscript as downloader
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--task", type=str, default='wic')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    task_name = args.task

    if task_name not in ['superglue_axb', 'superglue_axg']:

        if not os.path.exists(f"{args.data_dir}/tasks/data/{task_name}"):
            downloader.download_data([task_name], f"{args.data_dir}/tasks")

        if not os.path.exists(f"{args.data_dir}/models/{args.model_name}"):
            export_model.export_model(
                hf_pretrained_model_name_or_path=args.model_name_or_path,
                output_base_path=f"{args.data_dir}/models/{args.model_name}",
            )

        if not os.path.exists(f"{args.data_dir}/cache/{task_name}"):
            tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
                task_config_path=f"{args.data_dir}/tasks/configs/{task_name}_config.json",
                hf_pretrained_model_name_or_path=args.model_name_or_path,
                output_dir=f"{args.data_dir}/cache/{task_name}",
                phases=["train", "val", "test"],
            ))

        row = caching.ChunkedFilesDataCache(f"{args.data_dir}/cache/{task_name}/train").load_chunk(0)[0]["data_row"]

        jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
            task_config_base_path=f"{args.data_dir}/tasks/configs",
            task_cache_base_path=f"{args.data_dir}/cache",
            train_task_name_list=[task_name],
            val_task_name_list=[task_name],
            test_task_name_list=[task_name],
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            epochs=args.epoch,
            num_gpus=1,
        ).create_config()
        os.makedirs(f"{args.data_dir}/run_configs/{task_name}", exist_ok=True)
        py_io.write_json(jiant_run_config, f"{args.data_dir}/run_configs/{task_name}/{args.run_name}_run_config.json")
        display.show_json(jiant_run_config)

        run_args = main_runscript.RunConfiguration(
            jiant_task_container_config_path=f"{args.data_dir}/run_configs/{task_name}/{args.run_name}_run_config.json",
            output_dir=f"{args.data_dir}/runs/{args.run_name}/{task_name}",
            hf_pretrained_model_name_or_path=args.model_name_or_path,
            model_path=f"{args.data_dir}/models/{args.model_name}/model/model.p",
            model_config_path=f"{args.data_dir}/models/{args.model_name}/model/config.json",
            learning_rate=args.learning_rate,
            eval_every_steps=10000,
            do_train=True,
            do_val=True,
            do_save=True,
            force_overwrite=True,
            write_test_preds=True,
            write_val_preds=True
        )
        main_runscript.run_loop(run_args)
    else:
        if not os.path.exists(f"{args.data_dir}/tasks/data/{task_name}"):
            downloader.download_data([task_name], f"{args.data_dir}/tasks")

        if not os.path.exists(f"{args.data_dir}/models/{args.model_name}"):
            export_model.export_model(
                hf_pretrained_model_name_or_path=args.model_name_or_path,
                output_base_path=f"{args.data_dir}/models/{args.model_name}",
            )

        if not os.path.exists(f"{args.data_dir}/cache/{task_name}"):
            tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
                task_config_path=f"{args.data_dir}/tasks/configs/{task_name}_config.json",
                hf_pretrained_model_name_or_path=args.model_name_or_path,
                output_dir=f"{args.data_dir}/cache/{task_name}",
                phases=["test"],
            ))

        row = caching.ChunkedFilesDataCache(f"{args.data_dir}/cache/{task_name}/test").load_chunk(0)[0]["data_row"]

        jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
            task_config_base_path=f"{args.data_dir}/tasks/configs",
            task_cache_base_path=f"{args.data_dir}/cache",
            test_task_name_list=[task_name],
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            epochs=args.epoch,
            num_gpus=1,
        ).create_config()


        os.makedirs(f"{args.data_dir}/run_configs/{task_name}", exist_ok=True)
        py_io.write_json(jiant_run_config, f"{args.data_dir}/run_configs/{task_name}/{args.run_name}_run_config.json")
        display.show_json(jiant_run_config)

        run_args = main_runscript.RunConfiguration(
            jiant_task_container_config_path=f"{args.data_dir}/run_configs/{task_name}/{args.run_name}_run_config.json",
            output_dir=f"{args.data_dir}/runs/{args.run_name}/{task_name}",
            hf_pretrained_model_name_or_path=args.model_name_or_path,
            model_path=f"{args.data_dir}/models/{args.model_name}/model/model.p",
            model_config_path=f"{args.data_dir}/models/{args.model_name}/model/config.json",
            learning_rate=args.learning_rate,
            eval_every_steps=500,
            do_train=False,
            do_val=False,
            do_save=True,
            force_overwrite=True,
            write_test_preds=True,
            write_val_preds=False
        )
        main_runscript.run_loop(run_args)
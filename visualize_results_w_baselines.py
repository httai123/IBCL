from visualize_results import *


# Plot metrics of GEM + VCL + IBCL + FS
def plot_metrics_w_baselines(dict_gem, dict_gem_reg, dict_vcl, dict_vcl_reg, dict_ibcl, dict_fs_ibcl, task_nums=10, metric_name='Avg per task accuracy', bt=False):

    dicts = [dict_vcl, dict_vcl_reg, dict_ibcl, dict_fs_ibcl, dict_gem, dict_gem_reg]
    curve_colors = ['orange', 'green', 'blue', 'purple', 'darkgoldenrod', 'darkslategrey']
    fill_colors = ['gold', 'limegreen', 'tab:blue']

    if bt:
        task_range = list(range(1, task_nums))
        plot_range = list(range(task_nums - 1))
        plt.hlines(y=0.0, xmin=2, xmax=task_nums, colors='red', linestyles='-')
    else:
        task_range = list(range(task_nums))
        plot_range = list(range(task_nums))

    task_range = [int(j) for j in task_range]
    plot_range = [int(j) for j in plot_range]

    plot_data_gem = [dicts[4][j] for j in task_range if dicts[4] is not None]
    plot_data_gem_reg = [dicts[5][j] for j in task_range if dicts[5] is not None]

    plot_data_vcl = [dicts[0][j] for j in task_range if dicts[0] is not None]
    plot_data_vcl_reg = [dicts[1][j] for j in task_range if dicts[1] is not None]
    plot_data_ibcl = [dicts[2][j] for j in task_range if dicts[2] is not None]
    plot_data_fs_ibcl = [dicts[3][j] for j in task_range if dicts[3] is not None]

    if plot_data_gem:
        data_gem = [plot_data_gem[j][0] for j in plot_range]
        plt.plot(np.array(task_range) + 1, data_gem, '-^', label='GEM', color=curve_colors[4], markersize=5)
    if plot_data_gem_reg:
        data_gem_reg = [plot_data_gem_reg[j][0] for j in plot_range]
        plt.plot(np.array(task_range) + 1, data_gem_reg, '-^', label='GEM reg', color=curve_colors[5], markersize=5)

    if plot_data_vcl:
        max_data_vcl = [plot_data_vcl[j][0] for j in plot_range]
        plt.plot(np.array(task_range) + 1, max_data_vcl, '-o', label='VCL', color=curve_colors[0], markersize=5)
    if plot_data_vcl_reg:
        max_data_vcl_reg = [plot_data_vcl_reg[j][0] for j in plot_range]
        plt.plot(np.array(task_range) + 1, max_data_vcl_reg, '-o', label='VCL reg', color=curve_colors[1], markersize=5)
    if plot_data_ibcl:
        max_data_ibcl = [np.amax(plot_data_ibcl[j]) for j in plot_range]
        plt.plot(np.array(task_range) + 1, max_data_ibcl, '-*', label='IBCL', color=curve_colors[2], markersize=8)
    if plot_data_fs_ibcl:
        max_data_fs_ibcl = [np.amax(plot_data_fs_ibcl[j]) for j in plot_range]
        plt.plot(np.array(task_range) + 1, max_data_fs_ibcl, '-*', label='FS IBCL', color=curve_colors[3], markersize=8)

    if bt:
        plt.ylim([-0.2, 0.2])
    else:
        plt.ylim([0.0, 1.0])
    if task_nums <= 20:
        plt.xticks(np.array(task_range) + 1)
    else:
        plt.xticks(np.array(task_range[::5]) + 1)
    plt.xlabel('Task num')
    plt.ylabel(metric_name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--alpha", help="alpha value of IBCL in (0, 1)", default=0.5)
    parser.add_argument("--discard_threshold",
                        help="threshold on posterior param distances to discard new posteriors for sublinear buffer growth",
                        default=0.01)
    parser.add_argument("--module", help="specify module to run: gem, vcl, or all", default='all')
    parser.add_argument("--sublinear", help="sublinear flag", type=int, default=0)

    args = parser.parse_args()

    if args.task_name == 'cifar10':
        task_nums = 5
    elif args.task_name == 'celeba':
        task_nums = 15
    elif args.task_name == 'cifar100':
        task_nums = 10
    elif args.task_name == 'tinyimagenet':
        task_nums = 10
    elif args.task_name == '20newsgroup':
        task_nums = 5
    else:
        raise NotImplementedError

    # Load dict prefs
    dict_prefs = torch.load(os.path.join(args.data_dir, 'dict_prefs.pt'))

    # Load dict of accs
    dict_all_accs_gem = torch.load(os.path.join(args.data_dir, f'dict_all_accs_gem.pt'))
    dict_all_accs_gem[0] = [dict_all_accs_gem[0]]
    for i in range(1, task_nums):
        dict_all_accs_gem[i] = [dict_all_accs_gem[i]] * 10  # times number of prefs
    dict_all_accs_gem_reg = torch.load(os.path.join(args.data_dir, f'dict_all_accs_gem_reg.pt'))
    dict_all_accs_vcl = torch.load(os.path.join(args.data_dir, f'dict_all_accs_vcl.pt'))
    dict_all_accs_vcl_reg = torch.load(os.path.join(args.data_dir, f'dict_all_accs_vcl_reg.pt'))
    if float(args.discard_threshold) == 0.0:
        dict_all_accs_ibcl = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}.pt'))
    else:
        dict_all_accs_ibcl = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}_{args.discard_threshold}.pt'))
    dict_all_accs_fs_ibcl = torch.load(os.path.join(args.data_dir, f'dict_fs_all_few_accs_{args.alpha}_{args.discard_threshold}.pt'))

    # Compute metrics
    dict_matrix_gem = generate_acc_matrix(dict_all_accs_gem, dict_prefs, task_nums=task_nums, deterministic=True)
    dict_matrix_gem_reg = generate_acc_matrix(dict_all_accs_gem_reg, dict_prefs, task_nums=task_nums, deterministic=True)
    dict_matrix_vcl = generate_acc_matrix(dict_all_accs_vcl, dict_prefs, task_nums=task_nums, deterministic=False)
    dict_matrix_vcl_reg = generate_acc_matrix(dict_all_accs_vcl_reg, dict_prefs, task_nums=task_nums, deterministic=False)
    dict_matrix_ibcl = generate_acc_matrix(dict_all_accs_ibcl, dict_prefs, task_nums=task_nums, deterministic=False)
    dict_matrix_fs_ibcl = generate_acc_matrix(dict_all_accs_fs_ibcl, dict_prefs, task_nums=task_nums, deterministic=False)

    dict_avg_accs_gem = avg_per_task_acc(dict_matrix_gem, task_nums=task_nums, deterministic=True)
    dict_peak_accs_gem = peak_per_task_acc(dict_matrix_gem, task_nums=task_nums, deterministic=True)
    dict_avg_bts_gem = avg_bt(dict_matrix_gem, task_nums=task_nums, deterministic=True)

    dict_avg_accs_gem_reg = avg_per_task_acc(dict_matrix_gem_reg, task_nums=task_nums, deterministic=True)
    dict_peak_accs_gem_reg = peak_per_task_acc(dict_matrix_gem_reg, task_nums=task_nums, deterministic=True)
    dict_avg_bts_gem_reg = avg_bt(dict_matrix_gem_reg, task_nums=task_nums, deterministic=True)

    dict_avg_accs_vcl = avg_per_task_acc(dict_matrix_vcl, task_nums=task_nums)
    dict_peak_accs_vcl = peak_per_task_acc(dict_matrix_vcl, task_nums=task_nums)
    dict_avg_bts_vcl = avg_bt(dict_matrix_vcl, task_nums=task_nums)

    dict_avg_accs_vcl_reg = avg_per_task_acc(dict_matrix_vcl_reg, task_nums=task_nums)
    dict_peak_accs_vcl_reg = peak_per_task_acc(dict_matrix_vcl_reg, task_nums=task_nums)
    dict_avg_bts_vcl_reg = avg_bt(dict_matrix_vcl_reg, task_nums=task_nums)

    dict_avg_accs_ibcl = avg_per_task_acc(dict_matrix_ibcl, task_nums=task_nums)
    dict_peak_accs_ibcl = peak_per_task_acc(dict_matrix_ibcl, task_nums=task_nums)
    dict_avg_bts_ibcl = avg_bt(dict_matrix_ibcl, task_nums=task_nums)

    dict_avg_accs_fs_ibcl = avg_per_task_acc(dict_matrix_fs_ibcl, task_nums=task_nums)
    dict_peak_accs_fs_ibcl = peak_per_task_acc(dict_matrix_fs_ibcl, task_nums=task_nums)
    dict_avg_bts_fs_ibcl = avg_bt(dict_matrix_fs_ibcl, task_nums=task_nums)

    if args.module == 'gem' or args.module == 'all':
        plot_metrics_w_baselines(dict_avg_accs_gem, dict_avg_accs_gem_reg, None, None, dict_avg_accs_ibcl, dict_avg_accs_fs_ibcl, task_nums=task_nums, metric_name='avg per task acc', bt=False)
        plot_metrics_w_baselines(dict_peak_accs_gem, dict_peak_accs_gem_reg, None, None, dict_peak_accs_ibcl, dict_peak_accs_fs_ibcl, task_nums=task_nums, metric_name='peak per task acc', bt=False)
        plot_metrics_w_baselines(dict_avg_bts_gem, dict_avg_bts_gem_reg, None, None, dict_avg_bts_ibcl, dict_avg_bts_fs_ibcl, task_nums=task_nums, metric_name='avg per task bt', bt=True)

    if args.module == 'vcl' or args.module == 'all':
        plot_metrics_w_baselines(None, None, dict_avg_accs_vcl, dict_avg_accs_vcl_reg, dict_avg_accs_ibcl, dict_avg_accs_fs_ibcl, task_nums=task_nums, metric_name='avg per task acc', bt=False)
        plot_metrics_w_baselines(None, None, dict_peak_accs_vcl, dict_peak_accs_vcl_reg, dict_peak_accs_ibcl, dict_peak_accs_fs_ibcl, task_nums=task_nums, metric_name='peak per task acc', bt=False)
        plot_metrics_w_baselines(None, None, dict_avg_bts_vcl, dict_avg_bts_vcl_reg, dict_avg_bts_ibcl, dict_avg_bts_fs_ibcl, task_nums=task_nums, metric_name='avg per task bt', bt=True)

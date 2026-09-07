"""Build reviewed bounded report input; does not query a live collection or train a model."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from report_helpers import chart, chart_block, markdown, table, table_block

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
connection = sqlite3.connect(":memory:")
connection.create_function("readfile", 1, lambda path: (PROJECT / path).read_text())
NUMERIC_READ_SQL = """SELECT
  readfile('artifacts/distribution_audit_20260907/summary.json') AS summary_json,
  readfile('artifacts/distribution_audit_20260907/episodes.json') AS episodes_json;
"""
summary_json, episodes_json = connection.execute(NUMERIC_READ_SQL).fetchone()
summary = json.loads(summary_json)
episodes = json.loads(episodes_json)
proof = json.loads((OUT / "schema-proof.json").read_text())
visual = json.loads((OUT / "visual-summary.json").read_text())
visual_episodes = json.loads((OUT / "visual_episode_metrics.json").read_text())
snap = summary["snapshot"]
N = snap["complete_episodes_analyzed"]
assert len(episodes) == N
rates = summary["episode_rates"]
frates = summary["frame_rates"]
q = summary["trajectory_quantiles"]
timestamp = snap["snapshot_utc"]
relative_dataset = str(Path(snap["dataset"]).relative_to(PROJECT))
source_id = "distribution_snapshot"
report_dir = "artifacts/distribution_audit_20260907"


def pct(key):
    return rates[key]["percent"]


def entry(name, stat, definition):
    return {"event": name, "fraction": stat["fraction"], "count": stat["count"],
            "denominator": stat["denominator"], "definition": definition,
            "snapshot_version": snap["lance_version"]}


dataset_source = {
    "id": source_id,
    "label": f"PickCube 带噪启发式采样 · Lance v{snap['lance_version']}",
    "path": f"{report_dir}/summary.json",
    "query": {
        "engine": "SQLite3 / JSON1; readfile UDF registered in build_report_artifact.py",
        "language": "sql",
        "sql": NUMERIC_READ_SQL,
        "id": f"pickcube-distribution-lance-{snap['lance_version']}",
        "executed_at": timestamp,
        "description": "派生报告读取层：此实际执行SQL只读取已经审计的summary/episodes JSON。原始Lance到指标的计算由scripts/analyze_sampling_distribution.py完成，单独保留该源文件身份；展示聚合见build_report_artifact.py。",
        "tables_used": [f"{report_dir}/summary.json", f"{report_dir}/episodes.json", relative_dataset],
        "filters": [f"Lance version = {snap['lance_version']}", "完整回合：每回合200帧，step_idx恰为0..199", "状态统计使用全部完整帧；动作/转移统计排除每回合step_idx=199", "本报告为固定快照，不随后台采集自动刷新"],
        "metric_definitions": {
            "曾抬起": "回合内 object_height_above_table >= 0.05m 至少一次；不等同于 contact grasp。",
            "曾原生成功": summary["definitions"]["recomputed_native_success"],
            "成功前回桌": summary["definitions"]["table_return"] + "; occurrence before any native success in that episode",
            "动作能量": "先逆TCP旋转回base/world轴，按原控制器限制旋转向量范数并缩放到弧度，再逐轴平方求和/总和。",
            "散点抽样": "在按episode_idx排序的全部完整回合中，取700个等间隔索引；不按结果筛选。",
        },
    },
}
schema_source = {
    "id": "schema_proof", "label": f"状态布局与时序对齐交叉检查 · Lance v{proof['snapshot_version']}",
    "path": f"{report_dir}/schema-proof.json",
    "query": {"engine": "Python / NumPy / LanceDB", "language": "python",
              "description": "20个均匀抽样回合，验证序列化sim_state布局、坐标变换、原生success谓词和动作排列。该辅助样本与主统计版本分别标记。",
              "tables_used": [proof["dataset"]],
              "filters": [f"Lance version = {proof['snapshot_version']}", "20个均匀抽样episode，共4000帧"]},
}
visual_source = {
    "id": "pointcloud_visibility", "label": "实际1200点XYZ+RGB输入的随机回合抽查",
    "path": f"{report_dir}/visual-summary.json",
    "query": {"engine": "SQLite3 / JSON1; readfile UDF registered in build_report_artifact.py", "language": "sql",
        "executed_at": visual["snapshot"]["audit_utc"],
        "description": "派生报告聚合层：以下SQL把visual_episode_metrics.json中逐帧方块点计数按100点宽度分箱。上游audit_pointcloud_visibility.py读取实际pointcloud，以颜色及记录方块中心4.5cm邻域联合计数，原始计算文件另列为来源。",
        "tables_used": [f"{report_dir}/visual_episode_metrics.json", relative_dataset],
        "filters": [f"Lance version = {visual['snapshot']['lance_version']}",
            f"从{visual['snapshot']['population_episodes']}个完整回合中均匀无放回抽32个，seed={visual['snapshot']['sample_seed']}",
            "保留每回合全部200帧；共6400帧，不按成败筛选"],
        "metric_definitions": {"方块候选点": "点满足红色阈值，且在转成点云坐标系后的记录方块中心4.5cm以内；对宽松/严格阈值分别验证。", "零候选": "本代理未检测到方块点，不等同物理不可见或任务失败。"}},
}

never_success = N - rates["any_recomputed_native_success"]["count"]
datasets = {}
datasets["headlines"] = [{"episodes": N,
    "lift_fraction": rates["ever_elevated_5cm"]["fraction"],
    "ever_success_fraction": rates["any_recomputed_native_success"]["fraction"],
    "return_before_success_fraction": rates["returns_to_table_before_first_success"]["fraction"],
    "yaw_fraction": summary["actions"]["world_aligned_rotation_energy_fraction_xyz"][2],
}]
datasets["episode_events"] = [
    entry("曾抬高至少5cm", rates["ever_elevated_5cm"], "相对静置高度>=5cm至少一帧"),
    entry("曾到达目标", rates["ever_at_goal"], "cube-goal距离<=2.5cm至少一帧"),
    entry("曾满足原生成功", rates["any_recomputed_native_success"], "同帧到目标且机械臂关节速度符合静止阈值"),
    entry("成功前曾回到桌面", rates["returns_to_table_before_first_success"], "抬高>=5cm后降至<=2cm，且此前未原生成功"),
    {"event": "整回合从未原生成功", "fraction": never_success / N, "count": never_success,
     "denominator": N, "definition": "回合内所有帧均不满足重算原生success", "snapshot_version": snap["lance_version"]},
    entry("末帧仍满足原生成功", rates["final_frame_native_success"], "仅回合最后一帧的重算原生success"),
]


def hist_rows(key, *, scale=1, denominator=None):
    data = summary["histograms"][key]
    den = denominator if denominator is not None else sum(r["count"] for r in data)
    return [{"bin": f"{r['left']*scale:g}–{r['right']*scale:g}",
             "left": r["left"]*scale, "right": r["right"]*scale,
             "count": r["count"], "fraction": r["count"]/den,
             "denominator": den, "snapshot_version": snap["lance_version"]} for r in data]


datasets["first_lift_hist"] = hist_rows("first_lift_seconds")
datasets["max_height_hist"] = hist_rows("maximum_cube_height_m", scale=100, denominator=N)
datasets["goal_height_hist"] = hist_rows("initial_goal_height_m", scale=100, denominator=N)
excursions = Counter(r["elevation_excursions"] for r in episodes)
datasets["excursions"] = [{"times": str(k), "count": excursions[k], "fraction": excursions[k]/N,
    "denominator": N, "definition": "5cm进入、2cm退出的迟滞计数；非精确抓取重试次数"} for k in range(max(excursions)+1)]
sample_indices = sorted(set(round(i * (N-1) / 699) for i in range(700)))
datasets["initial_cube_positions"] = [{
    "episode_idx": episodes[i]["episode_idx"], "seed": episodes[i]["seed"],
    "x_cm": episodes[i]["initial_cube_world_xyz"][0]*100,
    "y_cm": episodes[i]["initial_cube_world_xyz"][1]*100,
    "initial_goal_z_cm": episodes[i]["initial_goal_world_xyz"][2]*100,
    "ever_success": "曾成功" if episodes[i]["any_recomputed_native_success"] else "未成功",
    "first_lift_s": episodes[i]["first_elevation_seconds"], "sample_denominator": N,
    "shown_episodes": len(sample_indices)} for i in sample_indices]
datasets["rotation_energy"] = [{"axis": axis, "fraction": float(value),
    "valid_actions": snap["paired_state_actions_analyzed"]} for axis, value in zip(
        ["世界X轴（roll）", "世界Y轴（pitch）", "世界Z轴（yaw）"],
        summary["actions"]["world_aligned_rotation_energy_fraction_xyz"])]
datasets["order_groups"] = [{"group": f"第{r['quarter']}组", "episodes": r["episodes"],
    "success_count": r["success"]["count"], "ever_success_fraction": r["success"]["fraction"],
    "lift_fraction": r["elevation"]["fraction"], "return_before_success_fraction": r["return_before_success"]["fraction"]}
    for r in summary["episode_order_quartiles"]]
datasets["episode_preview"] = [{"episode": r["episode_idx"], "seed": r["seed"],
    "ever_success": "是" if r["any_recomputed_native_success"] else "否",
    "first_lift_s": r["first_elevation_seconds"], "first_success_s": r["first_success_seconds"],
    "max_height_cm": round(r["cube_max_height_above_table_m"]*100, 2),
    "table_return_before_success": "是" if r["returns_to_table_before_first_success"] else "否",
    "elevation_excursions": r["elevation_excursions"]} for r in episodes[:10]]
pointcounts = [n for r in visual_episodes for n in r["thresholds"]["strict"]["cube_candidate_points_by_step"]]
assert len(pointcounts) == visual["snapshot"]["frames_analyzed"]
VISUAL_HIST_SQL = """WITH RECURSIVE
bins(left_edge) AS (SELECT 0 UNION ALL SELECT left_edge+100 FROM bins WHERE left_edge<1100),
points AS (
  SELECT CAST(p.value AS INTEGER) AS point_count
  FROM json_each(readfile('artifacts/distribution_audit_20260907/visual_episode_metrics.json')) e,
       json_each(e.value, '$.thresholds.strict.cube_candidate_points_by_step') p
)
SELECT b.left_edge, COUNT(p.point_count) AS frame_count,
       (SELECT COUNT(*) FROM points) AS denominator
FROM bins b LEFT JOIN points p
ON p.point_count>=b.left_edge AND p.point_count<b.left_edge+100
GROUP BY b.left_edge ORDER BY b.left_edge;
"""
visual_source["query"]["sql"] = VISUAL_HIST_SQL
datasets["point_counts"] = [{"bin": f"{left}–{left+99}", "count": count,
    "denominator": den, "episode_count": len(visual_episodes),
    "criterion": "严格颜色阈值且距记录cube中心<=4.5cm"}
    for left, count, den in connection.execute(VISUAL_HIST_SQL)]
assert sum(r["count"] for r in datasets["point_counts"]) == len(pointcounts)

title = "采样分布审计：首次抬起很熟练，状态与动作覆盖仍需补齐"
charts = [
    chart("event_rates", "完整回合中的行为事件比例", "episode_events", "event", "fraction",
          percent=True, x_label="事件", y_label="占全部完整回合的比例",
          subtitle=f"{N:,}个回合；这些事件可重叠，比例不应相加", tooltip=("count", "denominator", "definition")),
    chart("lift_timing", "首次抬高5cm的时间分布", "first_lift_hist", "bin", "count", chart_type="bar",
          x_label="首次抬高时间区间（秒）", y_label="回合数",
          subtitle=f"仅含{rates['ever_elevated_5cm']['count']:,}个曾抬起回合；每个时间箱宽0.5秒", tooltip=("count", "fraction", "denominator")),
    chart("max_heights", "每回合最大抬高高度分布", "max_height_hist", "bin", "count", chart_type="bar",
          x_label="相对静置高度的最大抬升（cm）", y_label="回合数",
          subtitle=f"全部{N:,}个回合；每箱5cm，覆盖的是高度这一边缘分布", tooltip=("count", "fraction", "denominator")),
    chart("initial_xy", "各回合首个保存状态的方块平面位置", "initial_cube_positions", "x_cm", "y_cm", chart_type="scatter",
          x_label="世界X（cm）", y_label="世界Y（cm）",
          subtitle=f"按回合编号等间隔抽{len(sample_indices)}／{N:,}个回合，未按成败筛选；首状态为第一次step之后",
          tooltip=("episode_idx", "seed", "initial_goal_z_cm", "ever_success", "first_lift_s")),
    chart("excursion_counts", "回合内抬起与回落的重复次数", "excursions", "times", "count", chart_type="bar",
          x_label="从<=2cm再次进入>=5cm的次数", y_label="回合数",
          subtitle=f"全部{N:,}个回合，显示0到{max(excursions)}次的完整尾部；主动释放也会计入", tooltip=("count", "fraction", "definition")),
    chart("cube_visibility", "实际输入中每帧的方块候选点数量", "point_counts", "bin", "count", chart_type="bar",
          source_id="pointcloud_visibility", x_label="方块候选点数量区间", y_label="帧数",
          subtitle="固定版本随机抽32回合、6400帧；每帧总计1200个XYZ+RGB点，连续帧并非独立样本",
          tooltip=("count", "denominator", "episode_count", "criterion")),
]

tables = [
    table("rotation_axes", "动作旋转能量的轴向分配", "rotation_energy",
        [{"field":"axis","label":"旋转轴"},{"field":"fraction","label":"平方能量占比","format":"percent"},
         {"field":"valid_actions","label":"有效动作数","format":"number"}],
        sort_by="fraction", descending=True,
        subtitle="经坐标逆变换及控制器范数裁剪后的物理旋转命令；数值差异不代表独立探索了该轴"),
    table("order_check", "按回合编号四等分的复核", "order_groups",
        [{"field":"group","label":"编号组"},{"field":"episodes","label":"回合数","format":"number"},
         {"field":"success_count","label":"曾成功回合","format":"number"},
         {"field":"ever_success_fraction","label":"曾成功比例","format":"percent"},
         {"field":"lift_fraction","label":"曾抬起比例","format":"percent"},
         {"field":"return_before_success_fraction","label":"成功前回桌比例","format":"percent"}], sort_by="group",
        subtitle="初步检查前后段分布；编号组不是随机实验组，不能推导因果关系"),
    table("preview", "前10个完整回合的可复查数据", "episode_preview",
        [{"field":"episode","label":"回合"},{"field":"seed","label":"Seed"},
         {"field":"ever_success","label":"曾原生成功"},{"field":"first_lift_s","label":"首次抬起（s）"},
         {"field":"first_success_s","label":"首次成功（s）"},{"field":"max_height_cm","label":"最大抬高（cm）"},
         {"field":"table_return_before_success","label":"成功前回桌"},{"field":"elevation_excursions","label":"抬起次数"}],
        sort_by="episode", subtitle="用于核对回合粒度与列含义；总体结论来自全快照，不来自这10条预览"),
]
cards = [{"id": k, "dataset": "headlines", "sourceId": source_id,
    "description": desc, "metrics": [{"field": field, "label": label, "format": fmt}]}
    for k, field, label, fmt, desc in [
        ("episodes_card", "episodes", "完整回合", "compact", "固定Lance版本内的全部完整200帧回合"),
        ("lift_card", "lift_fraction", "曾抬高至少5cm", "percent", "描述几何抬起，不等同于接触抓稳或完整任务成功"),
        ("success_card", "ever_success_fraction", "曾满足原生成功", "percent", "同帧到目标且机械臂符合静止条件；已与reward=1逐帧核对"),
        ("return_card", "return_before_success_fraction", "成功前曾回到桌面", "percent", "抬高5cm后回落到2cm以内，之前尚未成功；不能区分主动释放与滑落"),
    ]]

blocks = [
    markdown("title", "# " + title),
    markdown("answer", f"## 你的直觉有依据：首次抬起很集中，但整段轨迹并非全程顺利\n\n在这份已落盘快照中，{pct('ever_elevated_5cm'):.2f}%的回合曾把方块抬高至少5cm，首次抬起的中位时间只有{q['first_lift_seconds']['median']:.2f}秒；因此视频里夹起很流畅并不意外。同时，{pct('any_recomputed_native_success'):.2f}%曾达到完整原生成功条件，仍有{never_success}个回合从未成功，{pct('returns_to_table_before_first_success'):.2f}%在首次成功之前发生过回桌过程。\n\n这是一套以熟练抓取为主、有搬运和回落变化的任务数据。最明显的覆盖缺口是抓取始终失败的情况极少、姿态动作几乎只绕竖直轴旋转。它可作为世界模型训练数据；是否足以支撑冻结编码器后的PPO，需要训练后的独立评测，当前统计不能保证。", source_id),
    {"id":"headline_metrics","type":"metric-strip","cardIds":[x["id"] for x in cards]},
    markdown("scope", f"## 先区分回合、帧和成功的口径\n\n分析固定于Lance v{snap['lance_version']}，时间为{timestamp}，包含{N:,}个完整回合、{snap['states_analyzed']:,}帧。每回合200帧、间隔0.05秒；有效相邻状态—动作记录为{snap['paired_state_actions_analyzed']:,}条。\n\n“原生成功”重算自保存的方块—目标距离≤2.5cm，并且同帧七个机械臂关节的最大绝对速度≤0.2rad/s。“曾成功”指回合内至少一次，“末帧成功”只看最后一帧。“抬起”是相对静置高度增加至少5cm；TCP靠近、物体抬起都不能证明原生contact grasp。", source_id),
    chart_block("event_rates"),
    markdown("success_interpretation", f"## 达成过目标与最终保持目标是两种不同覆盖\n\n曾原生成功为{rates['any_recomputed_native_success']['count']:,}／{N:,}回合，但最后一帧仍成功仅{rates['final_frame_native_success']['count']}／{N:,}，即{pct('final_frame_native_success'):.2f}%。这反映当前采集包含成功之后继续运动、释放或转向的片段；低末帧比例不能倒推出整回合没有好样本。\n\n所有保存帧中，原生成功帧占{frates['recomputed_native_success']['percent']:.2f}%，物体抬高5cm的帧占{frates['elevated_5cm']['percent']:.2f}%。帧之间相关，这些比例描述数据占据时间，不是独立试验成功率。", source_id),
    markdown("easy_first_lift", f"## 抓起阶段的确偏熟练，少量视频会强化这种印象\n\n首次抬起发生时间的25%—75%分位为{q['first_lift_seconds']['p25']:.2f}—{q['first_lift_seconds']['p75']:.2f}秒，95%分位为{q['first_lift_seconds']['p95']:.2f}秒。整回合从未抬高5cm的只有{rates['never_elevated_5cm']['count']}个；因此数据很少展示“持续抓不到”的状态链。时间短只说明物体较快进入抬起状态，不等于无接触误差，也不能据此确定首次夹持成功率。", source_id),
    chart_block("lift_timing"),
    markdown("state_coverage", f"## 空间和高度有变化，但边缘范围不能证明组合覆盖充分\n\n每回合最大抬高的中位数为{q['max_cube_height_m']['median']*100:.1f}cm，5%—95%分位为{q['max_cube_height_m']['p05']*100:.1f}—{q['max_cube_height_m']['p95']*100:.1f}cm。首个保存状态的方块平面位置也有变化。下图用于查看分布形状；出现多个位置和高度，不意味着每种接近方向、姿态、接触状态及动作组合都被采到。", source_id),
    chart_block("max_heights"),
    chart_block("initial_xy"),
    markdown("returns", f"## 回落与重复抬起很多，其中一部分出现在成功以前\n\n{pct('returns_to_table_after_lift'):.2f}%的回合曾在抬起后回到桌面附近，{pct('multiple_elevation_excursions'):.2f}%出现不止一次抬起，重复次数中位数为{q['elevation_excursions']['median']:.0f}。其中{rates['returns_to_table_before_first_success']['count']:,}个回合在首次成功前已经发生回桌。\n\n这些是有实际物体运动的过程变化，值得用于检查失败和恢复片段。这里采用5cm进入、2cm退出的迟滞计数，避免阈值附近微小抖动；主动放下、释放后掉落与意外滑落仍可能混在一起，不能把每次回落都命名为一次失败或重试。", source_id),
    chart_block("excursion_counts"),
    markdown("action_gap", f"## 更确定的缺口是独立姿态动作覆盖\n\n把保存动作还原到统一轴向并应用控制器缩放后，旋转命令平方能量的{summary['actions']['world_aligned_rotation_energy_fraction_xyz'][2]*100:.2f}%落在世界yaw轴。启发式独立控制的是XYZ、yaw和夹爪，映射为七维动作并不增加独立的roll/pitch探索。出现的小量横向旋转主要与分量裁剪后的轴向失真有关。\n\n因此，全七维PPO在预训练分布外改变姿态时，世界模型预测器和冻结表征的表现都尚未验证。相比单纯增加失败比例，补充有目的的偏位接近、未夹稳、掉落恢复和独立姿态扰动，更直接对应当前缺失的状态—动作条件。", source_id),
    table_block("rotation_axes"),
    markdown("validation", "## 前后段统计接近，成功标签也能独立交叉检查\n\n按回合编号分成四组后，曾成功比例、曾抬起比例和成功前回桌比例相近，没有看到由少数早期回合单独决定总体结论的迹象。这不是随机对照实验，也没有为未来新数据作保证。主快照所有检查数值均有限，回合—步号没有重复；重算success与reward=1逐帧核对没有分歧。", source_id),
    table_block("order_check"),
    markdown("method", "## 缺失布尔列不再被错误当作全部失败\n\n当前Lance数据没有保存原生success、is_obj_placed、is_robot_static、object_grasped布尔列。前三项的所需位置和关节速度仍在sim_state里，因此可以重算并交叉检查；缺少接触冲量时，精确object_grasped不能重建。此前从缺失布尔列得到的零计数不适合评价这个数据集。\n\n状态为每步执行后的s1…s200；上游整理将动作移为a1…a199、a0。因此动作和下一状态统计剔除每回合末行，状态占据统计仍保留所有帧。主统计与辅助布局检查分别标记快照身份，避免混用分母。", source_id),
    table_block("preview"),
    markdown("limitations", "## 这些统计还不能回答接触质量或模型泛化\n\n当前结果是固定快照的描述统计。完整回合中的帧高度相关；不把按帧计算的二项区间当成独立样本不确定性。按回合给出的Wilson区间也依赖重置回合近似独立，仅适合辅助描述。\n\n几何接近不等于抓稳；回桌不等于意外失败；初次抬起迅速不等于后半程顺利。空间的边缘分布和动作总能量不能替代联合的状态—动作覆盖。点云抽查虽未发现方块完全丢失，仍未验证未见视角、遮挡和颜色变化的鲁棒性。世界模型尚未完成训练，本报告没有测试latent预测误差、长时滚动误差或冻结表征PPO表现。"),
    markdown("next_steps", "## 优先补充缺少的转移，再用分组评测判断是否够用\n\n保留当前成功搬运样本。先给已有轨迹分成正常接近/搬运、成功前回落、从未成功、长时间无物体运动等可复查组，并抽取对应点云片段审阅。若补采，优先增加带任务目的的接近偏差、短时夹持失败、失手后恢复与姿态扰动，同时留住足够成功转移；避免用纯随机乱动把有效交互稀释。\n\n训练之后，分别在熟练轨迹、失败/恢复轨迹、不同姿态和未见初态上检查预测误差与冻结表征PPO。需要继续确认：失败主要发生在哪一段；不同扰动是否覆盖同一状态下的不同动作；输入点云是否仍能分清物体、夹爪与接触附近变化。本报告只分析已有数据，没有修改正在运行的采样或训练。"),
]
visual_q = visual["thresholds"]["strict"]["point_count_quantiles"]
visual_at = next(i for i, block in enumerate(blocks) if block["id"] == "action_gap")
blocks[visual_at:visual_at] = [
    markdown("visual_inputs", f"## 实际点云抽查中，方块始终有可检测的点\n\n从同一版本随机抽32个回合，读取了6.4k帧实际1200点XYZ+RGB输入。用颜色阈值加真实方块位置邻域确认后，每帧方块候选点中位数为{visual_q['median']:.0f}，5%—95%分位为{visual_q['p05']:.0f}—{visual_q['p95']:.0f}，最少{visual_q['min']:.0f}。未出现零候选点或少于5点的帧；宽松和严格颜色阈值得到相同的最终计数。\n\n这说明该抽样中没有明显的物体点云丢失；不是对全量数据和遮挡鲁棒性的证明。候选点数量随距离与视角变化，不能据此推断接触质量或编码器已经学会物体动力学。", "pointcloud_visibility"),
    chart_block("cube_visibility"),
]

artifact = {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title,
    "generatedAt": timestamp, "description": "基于固定Lance快照的只读分布诊断；含指标定义、行为/空间/动作证据与模型泛化边界。",
    "blocks": blocks, "cards": cards, "charts": charts, "tables": tables,
    "sources": [dataset_source, schema_source, visual_source]},
    "snapshot": {"version": 1, "generatedAt": timestamp, "status": "ready", "datasets": datasets},
    "sources": [dataset_source, schema_source, visual_source]}

# Keep original transformations alongside the SQL presentation layer; no invented provenance.
upstream_sources = [
    {"id": "numeric_analysis_code", "label": "原始Lance数值分析代码", "path": "scripts/analyze_sampling_distribution.py"},
    {"id": "visual_analysis_code", "label": "原始Lance点云抽查代码", "path": f"{report_dir}/audit_pointcloud_visibility.py"},
    {"id": "report_transform_code", "label": "本报告的读取与展示聚合代码", "path": f"{report_dir}/build_report_artifact.py"},
]
artifact["manifest"]["sources"].extend(upstream_sources)
artifact["sources"].extend(upstream_sources)
for source in artifact["manifest"]["sources"]:
    definitions = source.get("query", {}).get("metric_definitions")
    if isinstance(definitions, dict):
        source["query"]["metric_definitions"] = [f"{k}: {v}" for k, v in definitions.items()]

assert all(len(rows) <= 2000 for rows in datasets.values())
assert sum(row["count"] for row in datasets["excursions"]) == N
assert sum(row["count"] for row in datasets["max_height_hist"]) == N
assert sum(row["count"] for row in datasets["goal_height_hist"]) == N
assert sum(row["count"] for row in datasets["first_lift_hist"]) == rates["ever_elevated_5cm"]["count"]
output = OUT / "artifact.json"
output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"artifact": str(output), "bytes": output.stat().st_size, "charts": len(charts), "blocks": len(blocks), "lance_version": snap["lance_version"]}, ensure_ascii=False))

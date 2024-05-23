import re

from typing import List
from pyecharts import options as opts
from pyecharts.charts import Line


def tolist(s: str) -> List[float]:
    numbers = s.split(",")
    ret = [float(num) for num in numbers]

    return ret


if __name__ == '__main__':
    file_name = "full/res_stack_se/log/24-05-23_01-28.log"
    # file_name = "full/res_stack_se/log/24-05-22_22-55.log"
    with open(file_name, encoding='utf8') as f:
        s = f.read()

    match = re.findall(r"DEBUG - \[(.*)]", s)
    if len(match) != 2:
        raise Exception(f"wrong match{match}")

    val, test = match
    val = tolist(val)
    test = tolist(test)
    diff = [abs(val_t - test_t) for val_t, test_t in zip(val, test)]
    line = Line()
    x_data = list(range(len(val) + 1))

    line.add_xaxis(x_data)
    line.add_yaxis("val", val, label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis("test", test, label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis("diff", diff, label_opts=opts.LabelOpts(is_show=False))

    # 设置全局配置项
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(is_show=True)
    )

    # 渲染图表到本地 HTML 文件
    line.render("line_chart.html")

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: XuXu

import numpy as np
import pandas as pd
import streamlit as st
import graphviz as gv
import plotly.express as px
import plotly.io as pio

pio.templates.default = 'plotly_white'

# 01 设定变量
# ------------------------------------------------------------------------------------------------------
path_result_train, path_result_predict = './results/result_train.xlsx', './results/result_predict.xlsx'
str_version_date = '2023-05-12'

# st.markdown("<p style='font-size: 15px; text-align: center; color: grey;'>图1&emsp;这是图标题</p>", unsafe_allow_html=True)

# 02-01 读取数据
# ------------------------------------------------------------------------------------------------------
# df_train = pd.read_excel(path_result_train, '01-df_train')
df_importances_dummy = pd.read_excel(path_result_train, '02-df_importances_dummy')
df_importances_origin = pd.read_excel(path_result_train, '03-df_importances_origin')
df_columns = pd.read_excel(path_result_train, '04-df_columns')
df_alpha = pd.read_excel(path_result_train, '05-df_info_alpha')
df_beta = pd.read_excel(path_result_train, '06-df_info_beta')

df_train = pd.read_excel(path_result_predict, '01-df_train')
df_train_x = pd.read_excel(path_result_predict, '02-df_train_x')
df_predict = pd.read_excel(path_result_predict, '03-df_predict')
df_predict_x = pd.read_excel(path_result_predict, '04-df_predict_x')
df_res = pd.read_excel(path_result_predict, '05-df_res')
df_res_long_tab = pd.read_excel(path_result_predict, '06-df_res_long_tab')
df_each_new_route = pd.read_excel(path_result_predict, '07-df_each_new_route')
df_new_routes_desc = pd.read_excel(path_result_predict, '08-df_new_routes_desc')

# 在原始数据的基础上 加工出来的一些变量
list_colnames_all = df_importances_origin.features_names.tolist()
list_colnames_discrete, list_colnames_continuous = list_colnames_all[:3], list_colnames_all[3:]

df_train_col_info = pd.DataFrame({
    '部分列名': [
        'α参数', 'β参数', 'α', 'β', 'alpha_predict', 'beta_predict',
        'leaf_alpha', 'path_alpha', 'leaf_beta', 'path_beta', 'path_str_alpha', 'path_str_beta'
    ],
    '对应的含义': [
        '原始模板数据中的α参数。', '原始模板数据中的β参数。',
        '经过四舍五入处理后的α参数。', '经过四舍五入处理后的β参数。',
        '训练好的模型对α列进行的预测。', '训练好的模型对β列进行的预测。',
        '推导α参数时落入的叶子节点编号。', '推导α参数时完整的路径编号序列。',
        '推导β参数时落入的叶子节点编号。', '推导β参数时完整的路径编号序列。',
        '把path_alpha的路径解码成自然语言。', '把path_beta的路径解码成自然语言。'
    ]
})


# 02-02 两阶段设计的流程图
# ------------------------------------------------------------------------------------------------------
gv_2step = gv.Digraph()

gv_2step.node('get_data', '历史数据', shape='folder')
gv_2step.node('bi', r'业\n务\n端|{1. 接收数据|2. 优化数据|3. 输出数据}', shape='record')
gv_2step.node('ai', r'数\n据\n端|{1. 数据建模|2. 输出模型|3. 生成数据}', shape='record')
gv_2step.node('predict', '预测衰变', shape='component')

gv_2step.edge('get_data', 'bi')
gv_2step.edge('bi', 'ai', style='dashed', label='提供建模数据')
gv_2step.edge('ai', 'bi', style='dashed', label='反馈生成数据')
gv_2step.edge('ai', 'predict')


# 02-03 循环优化的流程图
# ------------------------------------------------------------------------------------------------------
gv_circle = gv.Digraph()

with gv_circle.subgraph() as s:
    s.attr(rank='same')
    s.node('data_old', '历史数据', shape='folder')

with gv_circle.subgraph() as s:
    s.attr(rank='same')
    s.node('data_set', '数据库', shape='cylinder')

with gv_circle.subgraph() as s:
    s.attr(rank='same')
    s.node('route_old', '旧路段', shape='note')
    s.node('route_new', '新路段', shape='note')

gv_circle.node('model_set', '模型库', shape='cylinder')

gv_circle.edge('data_old', 'data_set', label='建库')
gv_circle.edge('data_set', 'route_old', label='取数', style='dashed')
gv_circle.edge('route_old', 'model_set', label='调优', style='dashed')
gv_circle.edge('model_set', 'route_new', label='预测', style='dashed')
gv_circle.edge('route_new', 'data_set', label='入库', style='dashed')


# 03 显示
# ------------------------------------------------------------------------------------------------------


# 简介
# ------------------------------------------------------------------------------------------------------
st.markdown(f'## 建模结果可视化-{str_version_date}')
st.markdown('---')

st.markdown('### 一、建模过程简介')

st.markdown('''
&emsp;&emsp;建模过程主要包括 **数据采集**、**数据处理**、**数据建模** 和 **模型应用** 四个步骤。
其中，数据采集是指，对历史路段特征进行 **采集**；
数据处理主要由 **业务端** 完成，它是指对相对粗糙的历史数据或生成数据进行过滤、加工或优化，使数据符合业务逻辑，并向数据端输出，以便其进行下一步分析建模；
数据建模主要由 **数据端** 完成，它根据业务端提供的数据进行建模，提取数据中的映射逻辑，并把训练完成的模型保存成计算机程序文件供预测使用。
除此以外，数据端还可以根据实际情况，利用训练完成的模型进行数据生成，并提供给业务端进行再次优化；
模型应用是指，调用数据端输出的模型文件，根据新路段的特征，对新路段进行 **衰变趋势** 的预测，辅助养护决策。

&emsp;&emsp;具体地，业务端收集 **经典路段数据集**，数据端根据此数据集建立 **路段特征对衰变参数** 的 ***CART*** 机器学习模型。
目前，已经基本实现了从数据采集到模型应用的第一阶段的开发，但是并未实现从数据端到业务端反馈强化的开发。
因此，后续可以在上述阶段的基础上，添加第二阶段 —— **数据反馈阶段**。此阶段的主要任务是，数据端根据业务端提供的 **边缘分布数据** 而建立的模型，生成 **联合分布数据**，并反馈给业务端。
业务端可以在此基础上，根据业务经验与业务逻辑，对生成的联合分布数据进行二次 **优化**，以此作为之前经典路段数据集的补充再次输出给数据端，接着再次启动第一阶段，以达到 **优化模型库** 的目的。

&emsp;&emsp;完整的两阶段设计可以实现 **业务端** 和 **数据端** 的交互，用 **业务知识** 指导 **数据建模**，用 **建模结果** 辅助 **业务决策**，
达到 **同时优化数据库和模型库** 的目的。
具体的流程可视化如下所示。
''')


# 部署两个图
# ------------------------------------------------------------------------------------------------------
with st.expander('📊 流程图：建模过程设计和系统迭代设计中 各个组件节点之间的关系'):
    col_1, col_2 = st.columns(2)

    with col_1:
        col_1.columns((0.5, 2.5))[1].graphviz_chart(gv_2step)

    with col_2:
        col_2.graphviz_chart(gv_circle)
        col_2.markdown('左图展示了建模步骤的 **交互迭代**')
        col_2.markdown('右图展示了系统应用的 **闭环设计**')


# 结果分析
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('### 二、结果分析')

st.markdown('''
&emsp;&emsp;结果分析主要包括两部分，分别是 **训练阶段** 结果分析和 **测试阶段** 结果分析。
其中，训练阶段结果主要包括 **训练集数据加工结果**、**特征重要性结果**、**列名编码结果** 和 **树模型结构结果**；
测试阶段结果主要包括 **测试数据加工结果** 和 **相似路段匹配结果**。
''')

st.markdown('''
&emsp;&emsp;作为信息的补充与具象化，会对各种表格型分析结果进行适当的 **可视化** 操作。
可视化操作的动机主要有 **四个**，分别是 **对离散型特征的探索性分析**、**对连续型特征的探索性分析**、**输入特征自变量对输出参数因变量的关系展示**，
以及 **重要结果的直观展示**。
''')


# 训练阶段
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('#### （一）训练阶段')


# 训练集数据加工结果
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('##### &emsp;1. 训练集数据加工结果')

st.markdown('''
&emsp;&emsp;此阶段主要是对训练数据的展示，其中包括了 **原始数据集**、**加工数据集**、**列名信息表** 以及 **可视化部分**。
原始数据集中尽量保留了 **原始特征**；加工数据集主要是对原始数据集中的特征进行 **筛选** 和 **转换** 等一系列操作后，所形成的能够直接用于建模的数据。
因此，如需复查数据，建议 **业务端** 从 **原始数据集** 进行切入，**数据端** 从 **加工数据集** 进行切入。
\n&emsp;&emsp;训练集数据加工结果如下所示。
''')

with st.expander('🧾 数据汇总表：训练集数据以及对其的加工结果'):
    list_tabs = st.tabs(['原始数据集', '加工数据集', '列名信息'])
    with list_tabs[0]:
        st.dataframe(df_train)
    with list_tabs[1]:
        st.dataframe(df_train_x)
    with list_tabs[2]:
        st.dataframe(df_train_col_info)

# 旭日图
fig_train_sunburst_alpha = px.sunburst(
    df_train, path=list_colnames_discrete, color='α参数', color_continuous_scale='RdYlBu_r',
    title='离散特征之间的包含关系及其对α参数的影响'
)
fig_train_sunburst_beta = px.sunburst(
    df_train, path=list_colnames_discrete, color='β参数', color_continuous_scale='RdYlBu_r',
    title='离散特征之间的包含关系及其对β参数的影响'
)

# 组合图
df_temp_alpha = df_train.loc[:, ['构造物', '养护措施', '结构层类型', 'α参数']]
df_temp_beta = df_train.loc[:, ['构造物', '养护措施', '结构层类型', 'β参数']]
df_temp_alpha.columns = ['构造物', '养护措施', '结构层类型', '参数值']
df_temp_beta.columns = ['构造物', '养护措施', '结构层类型', '参数值']
df_temp = pd.concat([df_temp_alpha, df_temp_beta])
del df_temp_alpha, df_temp_beta
df_temp['参数名称'] = np.repeat(['α参数值', 'β参数值'], len(df_train))

fig_train_violin_1 = px.violin(
    df_temp, y='参数值', x='构造物', color='参数名称', box=True, points='all',
    title='构造物对α参数和β参数的影响'
)
fig_train_violin_1.update_layout(yaxis=dict(type='log'), yaxis_range=[-2, 2])

fig_train_violin_2 = px.violin(
    df_temp, y='参数值', x='养护措施', color='参数名称', box=True, points='all',
    title='养护措施对α参数和β参数的影响'
)
fig_train_violin_2.update_layout(yaxis=dict(type='log'), yaxis_range=[-2, 2])

fig_train_violin_3 = px.violin(
    df_temp, y='参数值', x='结构层类型', color='参数名称', box=True, points='all',
    title='结构层类型对α参数和β参数的影响'
)
fig_train_violin_3.update_layout(yaxis=dict(type='log'), yaxis_range=[-2, 2])

# 热力图
df_temp = (
    df_train.
    loc[:, list_colnames_continuous + ['α参数', 'β参数']].
    corr()
)
fig_train_heatmap = px.imshow(
    df_temp, color_continuous_scale='RdYlBu_r', text_auto=True, color_continuous_midpoint=0, aspect='auto',
    title='连续特征和α参数以及β参数的相关热图  建议结合PCA相似性切入'
)

with st.expander('📊 数据可视化：离散型特征和连续型特征对双参数的影响'):
    list_tabs = st.tabs(['α旭日图', 'β旭日图', '构造物组合图', '养护措施组合图', '结构物类型组合图', '连续特征热力图'])
    with list_tabs[0]:
        st.plotly_chart(fig_train_sunburst_alpha, use_container_width=True)
    with list_tabs[1]:
        st.plotly_chart(fig_train_sunburst_beta, use_container_width=True)
    with list_tabs[2]:
        st.plotly_chart(fig_train_violin_1, use_container_width=True)
    with list_tabs[3]:
        st.plotly_chart(fig_train_violin_2, use_container_width=True)
    with list_tabs[4]:
        st.plotly_chart(fig_train_violin_3, use_container_width=True)
    with list_tabs[5]:
        st.plotly_chart(fig_train_heatmap, use_container_width=True)


st.markdown('''
&emsp;&emsp;值得强调的是，之前制作的相关热图是纳入了 **全量特征** 进行绘制，这次只纳入了 **连续型特性** 以及输出参数。
原因是，全量特征包括了离散型特征和连续型特征，**相关热图通常展示Pearson相关系数**，而**Pearson相关系数用于描述连续型特征**。
理论上，离散型特征有其专门的相关性进行计算。因此如有需要，可以按照同样的方式绘制离散型特征的相关热图，只是使用的相关系数并不是Pearson相关系数。
''')


# 特征重要性结果
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('##### &emsp;2. 特征重要性结果')
st.markdown('''
&emsp;&emsp;此阶段主要是对各个特征对双参输出重要性的展示，其中包括了 **转换前特征重要性** 和 **转换后特征重要性**。
特征的转换是对特征进行数值化处理，主要针对 **离散型特征**。转换的具体操作是，把单个离散特征的多个分类水平，通过 **独热编码** 的方式，转换成多个二分类特征。
其中，本来该二分类为 **阳性** 编码为1，**阴性** 编码为0。因此，转换后的特征个数多于原特征个数，
转换后特征个数等于扩增后离散型特征个数加上原连续型特征个数。
\n&emsp;&emsp;特征重要性结果如下所示。
''')

with st.expander('🧾 数据汇总表：特征重要性结果'):
    list_tabs = st.tabs(['转换前特征重要性', '转换后特征重要性'])
    with list_tabs[0]:
        st.dataframe(df_importances_origin)
    with list_tabs[1]:
        st.dataframe(df_importances_dummy)

df_temp = df_importances_origin.copy()
df_temp = df_temp.loc[:, ['features_names', 'importances_total', 'importances_alpha', 'importances_beta']]
df_temp.columns = ['特征名称', '重要性汇总', '重要性α', '重要性β']
df_temp = df_temp.sort_values(['重要性汇总', '重要性α', '重要性β'], ascending=False)
df_temp = pd.melt(df_temp, id_vars='特征名称', var_name='分组', value_name='重要性')
fig_importances_origin = px.bar(
    df_temp, x='特征名称', y='重要性', color='分组', barmode='group',
    title='原始特征重要性  离散特征重要性是变换后同一组特征的重要性总和'
)

df_temp = df_importances_dummy.copy()
df_temp = df_temp.loc[:, ['features_names', 'importances_total', 'importances_alpha', 'importances_beta']]
df_temp.columns = ['特征名称', '重要性汇总', '重要性α', '重要性β']
df_temp = df_temp.sort_values(['重要性汇总', '重要性α', '重要性β'], ascending=False)
df_temp = pd.melt(df_temp, id_vars='特征名称', var_name='分组', value_name='重要性')
fig_importances_dummy = px.bar(
    df_temp, x='特征名称', y='重要性', color='分组', barmode='group',
    title='离散特征经过了转换  如无其他特别需求  建议以此变换后的重要性进行切入'
)

with st.expander('📊 数据可视化：特征对双参数的影响'):
    list_tabs = st.tabs(['变换前特征的重要性', '转换后特征的重要性'])
    with list_tabs[0]:
        st.plotly_chart(fig_importances_origin, use_container_width=True)
    with list_tabs[1]:
        st.plotly_chart(fig_importances_dummy, use_container_width=True)


# 特征信息结果
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('##### &emsp;3. 列名编码结果')
st.markdown('''
&emsp;&emsp;此阶段主要是对各个特征编码的展示。
在建模的过程当中，会对输入特征进行编码，方便推导和描述。
这里提供特征和编号的对应信息表，以便进行 **特征名称** 和 **编号** 之间的 **查询** 和 **反查**。
\n&emsp;&emsp;特征编码结果如下所示。
''')

df_feature_col_info = pd.DataFrame({
    '列名': ['idx', 'columns'],
    '对应的含义': ['对应columns列的节点编号。', '对应idx的特征名称。']
})

with st.expander('🧾 数据汇总表：特征名称和编号的对应表格'):
    list_tabs = st.tabs(['特征编号', '列名信息'])
    with list_tabs[0]:
        st.dataframe(df_columns)
    with list_tabs[1]:
        st.dataframe(df_feature_col_info)


# 特征信息结果
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('##### &emsp;4. 树模型结构结果')
st.markdown('''
&emsp;&emsp;此阶段主要是对训练完成后的树模型结构进行展示。
预测参数有两个，分别是 **α参数** 和 **β参数**，分别对应 **α树** 和 **β树**。
树模型结构主要由节点和分裂逻辑组成。分裂逻辑主要包括节点分裂时对应的 **特征** 以及其 **分裂点取值**，还有向左向右流动后 **到达的节点**。
\n&emsp;&emsp;树模型结构结果如下所示。
''')

df_tree_col_info = pd.DataFrame({
    '列名': ['idx', 'feature', 'left', 'right', 'threshold'],
    '对应的含义': [
        '树模型中节点的编号。',
        '该节点根据特征进行分裂时，对应特征的编号。特征与编号的对应关系见上述的特征编号对应表。',
        '该节点分裂后的左节点编号。',
        '该节点分裂后的右节点编号。',
        '该节点进行分裂时，对应特征的分裂点取值。小于等于该取值往左流动，大于该取值往右流动。'
    ]
})

with st.expander('🧾 数据汇总表：树模型结构信息表格'):
    list_tabs = st.tabs(['α树信息', 'β树信息', '列名信息'])
    with list_tabs[0]:
        st.dataframe(df_alpha)
    with list_tabs[1]:
        st.dataframe(df_beta)
    with list_tabs[2]:
        st.dataframe(df_tree_col_info)


# 测试阶段
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('#### （二）测试阶段')


# 测试数据加工结果
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('##### &emsp;1. 测试数据加工结果')

st.markdown('''
&emsp;&emsp;此阶段主要是对测试数据的展示，其中包括了 **原始数据集**、**加工数据集**、**列名信息表** 以及 **可视化部分**。
各个部分跟训练阶段中同名部分相似。
测试集数据加工结果如下所示。
''')

df_predict_col_info = pd.DataFrame({
    '部分列名': [
        'alpha_predict', 'beta_predict',
        'leaf_alpha', 'path_alpha',
        'leaf_beta', 'path_beta',
        'path_str_alpha', 'path_str_beta'
    ],
    '对应的含义': [
        '根据特征预测的α参数。', '根据特征预测的β参数。',
        '推导α参数时落入的叶子节点编号。', '推导α参数时完整的路径编号序列。',
        '推导β参数时落入的叶子节点编号。', '推导β参数时完整的路径编号序列。',
        '把path_alpha的路径解码成自然语言。', '把path_beta的路径解码成自然语言。'
    ]
})
with st.expander('🧾 数据汇总表：测试集数据以及对其的加工结果'):
    list_tabs = st.tabs(['原始数据集', '加工数据集', '列名信息'])
    with list_tabs[0]:
        st.dataframe(df_predict)
    with list_tabs[1]:
        st.dataframe(df_predict_x)
    with list_tabs[2]:
        st.dataframe(df_predict_col_info)

# 树图
df_predict_temp = df_predict.copy()
df_train_temp = df_train.copy()
df_predict_temp['构造物'] = df_predict_temp['构造物'].fillna('路基')
df_predict_temp['养护措施'] = df_predict_temp['养护措施'].fillna('无')
df_predict_temp['数据集'] = np.repeat(['测试集'], len(df_predict_temp))
df_train_temp['数据集'] = np.repeat(['训练集'], len(df_train_temp))
df_temp = pd.concat([df_predict_temp, df_train_temp])
df_temp = df_temp.reset_index(drop=True)
del df_train_temp, df_predict_temp

fig_predict_treemap = px.treemap(
    df_temp,
    path=[px.Constant('全量数据'), '数据集', '构造物', '养护措施', '结构层类型'],
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)
fig_predict_treemap.update_traces(root_color='lightgreen')
fig_predict_treemap.update_layout(margin=dict(t=25, l=25, r=25, b=25))

fig_predict_hist_1 = px.histogram(
    df_temp, x='通车年限', color='数据集', marginal='box',
    color_discrete_sequence=['#EF553B', '#636EFA'], opacity=0.7,
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)
fig_predict_hist_2 = px.histogram(
    df_temp, x='设计弯沉（0.01mm）', color='数据集', marginal='box',
    color_discrete_sequence=['#EF553B', '#636EFA'], opacity=0.7,
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)
fig_predict_hist_3 = px.histogram(
    df_temp, x='路面总厚度（cm）', color='数据集', marginal='box',
    color_discrete_sequence=['#EF553B', '#636EFA'], opacity=0.7,
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)
fig_predict_hist_4 = px.histogram(
    df_temp, x='沥青层厚度（cm）', color='数据集', marginal='box',
    color_discrete_sequence=['#EF553B', '#636EFA'], opacity=0.7,
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)
fig_predict_hist_5 = px.histogram(
    df_temp, x='交通量（自然数）（辆/日）', color='数据集', marginal='box',
    color_discrete_sequence=['#EF553B', '#636EFA'], opacity=0.7,
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)
fig_predict_hist_6 = px.histogram(
    df_temp, x='三四五六类车（辆/日）', color='数据集', marginal='box',
    color_discrete_sequence=['#EF553B', '#636EFA'], opacity=0.7,
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)
fig_predict_hist_7 = px.histogram(
    df_temp, x='重车比例（%）', color='数据集', marginal='box',
    color_discrete_sequence=['#EF553B', '#636EFA'], opacity=0.7,
    title='建议从两个集合相似性切入  思考使用该训练集对测试集进行预测的科学性'
)

with st.expander('📊 数据可视化：测试集和训练集的特征分布对比'):
    list_tabs = st.tabs(['离散特征', '通车年限', '设计弯沉', '路面厚度', '沥青层厚度', '交通量', '三到六类车', '重车比重'])
    with list_tabs[0]:
        st.plotly_chart(fig_predict_treemap, use_container_width=True)
    with list_tabs[1]:
        st.plotly_chart(fig_predict_hist_1, use_container_width=True)
    with list_tabs[2]:
        st.plotly_chart(fig_predict_hist_2, use_container_width=True)
    with list_tabs[3]:
        st.plotly_chart(fig_predict_hist_3, use_container_width=True)
    with list_tabs[4]:
        st.plotly_chart(fig_predict_hist_4, use_container_width=True)
    with list_tabs[5]:
        st.plotly_chart(fig_predict_hist_5, use_container_width=True)
    with list_tabs[6]:
        st.plotly_chart(fig_predict_hist_6, use_container_width=True)
    with list_tabs[7]:
        st.plotly_chart(fig_predict_hist_7, use_container_width=True)


# 相似路段匹配结果
# ------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown('##### &emsp;2. 相似路段匹配结果')

st.markdown('''
&emsp;&emsp;此阶段主要是对每一段新路段进行经典路段的相似匹配，寻找出跟每一段新路段最相似的若干段经典路段。
相似路段匹配结果如下所示。
''')

df_sim_col_info = pd.DataFrame({
    '列名': [
        'idx_new_route',
        'idx_old_route_tree',
        'sim_tree',
        'idx_old_route_cos',
        'sim_cos',
        'idx_old_route_pca',
        'sim_pca',
        '序号_new',
        '序号_tree',
        '序号_cos',
        '序号_pca',
        '序号',
        'rbo_tree_vs_cos',
        'rbo_tree_vs_pca',
        'rbo_cos_vs_pca',
        'str_tree_vs_cos',
        'str_tree_vs_pca',
        'str_cos_vs_pca',
        'alpha_predict',
        'beta_predict'
    ],
    '对应的含义': [
        '该新路在测试数据表格中的第几行。',
        '基于树模型的相似性，找出对应的旧路在训练表格中的第几行。',
        '基于树模型相似性方法的相似性值。',
        '基于余弦相似性，找出对应的旧路在训练表格中的第几行。',
        '基于余弦相似性方法的相似性值。',
        '基于降维去相关后的余弦相似性，找出对应的旧路在训练表格中的第几行。',
        '基于降维去相关余弦相似性方法的相似性值。',
        '该新路在测试数据表格中的序号列的取值。',
        '基于树模型的相似性，找出对应的旧路在训练表格中序号列的取值。',
        '基于余弦相似性，找出对应的旧路在训练表格中序号列的取值。',
        '基于降维去相关后的余弦相似性，找出对应的旧路在训练表格中序号列的取值。',
        '该新路在测试数据表格中的序号列的取值。与前一个表格中的"序号_new"一致。',
        '该新路使用树方法和余弦相似性方法找出来的旧路排序的rbo值。',
        '该新路使用树方法和降维去相关后的余弦相似性方法找出来的旧路排序的rbo值。',
        '该新路使用余弦相似性方法和降维去相关后的余弦相似性方法找出来的旧路排序的rbo值。',
        'rbo_tree_vs_cos的分段标签化。',
        'rbo_tree_vs_pca的分段标签化。',
        'rbo_cos_vs_pca的分段标签化。',
        '该新路基于树模型预测出来的α参数，与测试数据加工的同名列一致。',
        '该新路基于树模型预测出来的β参数，与测试数据加工的同名列一致。'
    ]
})

with st.expander('🧾 数据汇总表：根据三种不同的相似性计算方法找出模板库中与测试新路最相似的路段'):
    list_tabs = st.tabs(['相似路段汇总', '不同相似性的结果一致性', '列名信息'])
    with list_tabs[0]:
        st.dataframe(df_res)
    with list_tabs[1]:
        st.dataframe(df_each_new_route)
    with list_tabs[2]:
        st.dataframe(df_sim_col_info)












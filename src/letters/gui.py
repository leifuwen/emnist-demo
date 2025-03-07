import operator
from nicegui import ui
from src.letters.letter_predict import answer_correct
from src.letters.process import split, answercrop


class Data:
    def __init__(self):
        self.num = 5
        self.soc = 0


data = Data()


def getimgpath(e):
    """
    上传事件函数
    :param e: 上传的数据
    :return:
    """
    img_bytes = e.content.read()
    with open('uploads/answer.png', 'wb') as f:
        f.write(img_bytes)
        f.close()
    # 裁剪答题卡
    answercrop()
    # 分割答题卡表格
    split()


with ui.element('div').style("margin:auto"):
    # 卡片组件
    with ui.card().style("height:100%;width:80vh"):
        # 上传组件
        ui.upload(on_upload=lambda e: getimgpath(e), auto_upload=True).props("accept=image/*").style(
            "width:100%").classes(
            'max-w-full')
        # 横向布局组件
        with ui.row().style("width:100%"):
            # 输入框组件,仅限数字，绑定数据来实时响应
            ui.input('每题分值(默认为5分)').bind_value_to(data, 'num').props('type=number')
            ui.button('评阅', on_click=lambda: click()).style('top:15px')
            ui.label().bind_text_from(data, "soc", backward=lambda n: f'得分：{n}').style("margin:auto")
        # 表格列数据
        columns = [
            {'name': 'id', 'label': '题号', 'field': 'id', 'sortable': True},
            {'name': 'answer', 'label': '正确答案', 'field': 'answer', 'sortable': True},
            {'name': 'my_answer', 'label': '你的答案', 'field': 'my_answer', 'sortable': True},
            {'name': 'is_True', 'label': '是否正确', 'field': 'is_True', 'sortable': True},
        ]
        rows = []
        # 表格占位
        for i in range(20):
            rows.append({})
        # 表格组件
        # 表格组件
        table = ui.table(columns=columns, rows=rows, row_key='id').style(
            "width:100%")


        def click():
            """
            评阅按钮的点击事件函数
            :return:
            """
            if data.num == '':
                data.num = 5
            global rows
            # 接受预测结果并排序
            rows = sorted(answer_correct(), key=operator.itemgetter('id'))
            # 更新表格数据
            table.update_rows(rows)
            score = 0
            # 计算分数
            for row in rows:
                if row['is_True']:
                    score += int(data.num)
            data.soc = score
            print(data.soc)
ui.run()

if __name__ == "__main__":
    print("开始")

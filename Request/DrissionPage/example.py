import os
import time
from time import sleep
from datetime import datetime, timedelta
from typing import List, Union
import pandas as pd
from DrissionPage import ChromiumPage, ChromiumOptions, errors
from retrying import retry
import logging

# logging.basicConfig(
#     encoding='utf-8',
#     level=logging.INFO,
#     format="[%(levelname)s]-[%(asctime)s] - { position: %(filename)s | func: %(funcName)s | line:%(lineno)d } -->> %(message)s",
#     datefmt='%H:%M:%S'
# )


def save_dict_list_to_csv(
        dict_list: List[dict], filename: str = 'weibo_data.csv'):
    df = pd.DataFrame(dict_list)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    logging.info('保存数据成功')


def is_within_last_30_days(date_string):
    # 定义日期格式
    date_format = "%Y-%m-%d"
    # 解析输入的日期字符串，返回有偏的datetime对象
    date_input = datetime.strptime(date_string, date_format)
    # 获取当前日期，返回无偏datetime对象
    current_date = datetime.now()
    # 计算30天前的日期，返回无偏datetime对象
    date_30_days_ago = current_date - timedelta(days=30)
    # 判断输入的日期是否在30天前和当前日期之间
    within_30_days = date_30_days_ago <= date_input.replace(
        tzinfo=None) <= current_date
    # 格式化当前日期时间字符串
    formatted_date_input = date_input.strftime("%Y-%m-%d %H:%M:%S")
    return within_30_days, formatted_date_input


class RedBook:
    def __init__(self, user_id: str = None, user_name: str = None,
                 headless=False, no_img=False):
        self.isLogin = False
        # 加载浏览器配置
        oc = ChromiumOptions()
        if no_img:
            oc.no_imgs()
        if headless:
            oc.headless()
        self.page = ChromiumPage(oc)
        # 初始化信息
        self.user_id = user_id
        self.user_name = user_name
        self.user_profile = {}
        self.res = []

    @retry(
        stop_max_attempt_number=3,
        wait_fixed=1000,
    )
    def wait_element(self, selector: str):
        flag = self.page.wait.eles_loaded(selector, timeout=0.3)
        if not flag:
            logging.info('等待超时，尝试重试')
            raise errors.WaitTimeoutError()

    def is_login(self):
        self.page.get('http://www.xiaohongshu.com/explore')
        lis = self.page.eles('css:ul.channel-list li')
        # logging.info(lis)
        count = len(lis)
        # logging.info(count)
        if count == 3:
            self.isLogin = False
            self.login()
        elif count == 4:
            self.isLogin = True
            logging.info(f'登录成功')
        else:
            logging.info(f'li 的数量不是 3 和 4 无法判定是否登录')

    def login(self):
        input('请扫码登录，登录成功后敲击回车')
        self.isLogin = True

    def check_user_id(self):
        if not self.user_id:
            if not self.user_name:
                raise ValueError('user_id 和 user_name 至少填写一个')
            else:
                self.page.ele('css:#search-input').input(self.user_name)
                self.page.actions.key_down('Enter')
                time.sleep(0.5)
                self.page.ele('css:.content-container').ele('用户').click()
                if self.page.eles('css:div.user-list-item'):
                    self.user_id = self.page.eles(
                        'css:div.user-list-item')[0].s_ele('css:a').attr('href').split('/')[-1]
                else:
                    raise ValueError('无匹配的用户')

    def to_user_page(self):
        url = f'http://www.xiaohongshu.com/user/profile/{self.user_id}'
        self.page.get(url)

    def get_user_profile(self):
        time.sleep(0.2)
        self.wait_element('css:.data-info .user-interactions div')
        data_div = self.page.s_eles('css:.data-info .user-interactions div')
        user_profile = {
            '用户名称': self.page.s_ele('css:.user-name').text,
            '用户ID': self.page.s_ele('css:.user-redId').text,
            '粉丝数': data_div[1].s_ele('css:.count').text,
            '获赞与收藏': data_div[2].s_ele('css:.count').text,
        }
        logging.info(f'用户简介：{user_profile}')
        self.user_profile = user_profile
        return user_profile

    def save_posts_to_csv(self, posts: list):
        # 创建导出文件夹
        if not os.path.exists('./output_data'):
            os.mkdir('output_data')
        # 加载用户数据
        if not self.user_profile:
            self.get_user_profile()
        df = pd.DataFrame(posts)
        filename = f'./output_data/red_book_{self.user_profile["用户名称"]}_{self.user_profile["用户ID"]}.csv'
        df.to_csv(filename, encoding='utf-8-sig', index=False)

    def get_all_post_detail(self):
        # ac = Actions(self.page)
        max_threshold = 3
        threshold = 0
        sections = self.page.s_eles('css:#userPostedFeeds section')
        max_index = int(sections[-1].attr('data-index'))
        logging.info(f'max_index: {max_index}')
        current_index = 0
        while current_index < max_index:
            # 点开详情页
            sleep(0.2)
            target = self.page.ele(f'css:*[data-index="{current_index}"]')
            cover = target.ele('css:.cover')
            cover.click()
            self.wait_element('c:div.close')
            sleep(0.2)
            # 提取数据
            date = self.page.s_ele(
                'css:div.note-scroller div.bottom-container span.date').text.strip().split(' ')[0]
            like_count = self.page.s_ele(
                'css:.engage-bar .like-wrapper span.count').text.strip()
            collect_count = self.page.s_ele(
                'css:.engage-bar .collect-wrapper span.count').text.strip()
            comment_count = self.page.s_ele(
                'css:.engage-bar .chat-wrapper span.count').text.strip()
            t_data = {
                '用户ID': self.user_profile['用户ID'],
                '用户名称': self.user_profile['用户名称'],
                '粉丝数': self.user_profile['粉丝数'],
                '获赞与收藏': self.user_profile['获赞与收藏'],
                "日期": date,
                "点赞数": like_count,
                "收藏数": collect_count,
                "评论数": comment_count,
            }
            # 做判断是否停止和舍弃
            flag = is_within_last_30_days(date)[0]
            logging.info(f'抓取数据：{t_data}，是否保存：{flag}')
            if not flag:
                # 如果超出目标区间，做阈值累加
                threshold += 1
                # 阈值超出，跳出循环，完成爬取
                if threshold > max_threshold:
                    break
            else:
                # 在目标区间，添加数据
                self.res.append(t_data)
            self.page.ele('css:.close').click()
            sections = self.page.s_eles('css:#userPostedFeeds section')
            max_index = int(sections[-1].attr('data-index'))
            # 处理偶发的 悬浮元素影响 数据提取的操作
            try:
                int(t_data['收藏数'])
            except BaseException:
                logging.info('获取收藏数失败，关闭当前详情页面重试')
                continue
            # 正常结束的收尾操作
            logging.info(
                f'current_index: {current_index} | max_index: {max_index}')
            current_index += 1
            sleep(0.2)

    @staticmethod
    def is_str_to_int(data: Union[str, int]) -> int:
        if not data:
            return 0
        if isinstance(data, str):
            return int(data)
        if isinstance(data, int):
            return data
        raise TypeError('类型错误')

    def sum_data(self):
        like_amount = 0
        collect_amount = 0
        chat_amount = 0
        for item in self.res:
            like_amount += self.is_str_to_int(item['点赞数'])
            collect_amount += self.is_str_to_int(item['收藏数'])
            chat_amount += self.is_str_to_int(item['评论数'])
        return {
            '用户ID': self.user_profile['用户ID'],
            '用户名称': self.user_profile['用户名称'],
            '粉丝数': self.user_profile['粉丝数'],
            '获赞与收藏': self.user_profile['获赞与收藏'],
            '点赞数': like_amount,
            '收藏数': collect_amount,
            '评论数': chat_amount,
            '近30天发文数': len(self.res)
        }

    @retry(
        stop_max_attempt_number=3,
        wait_fixed=1000,
    )
    def run(self):
        self.is_login()
        self.check_user_id()
        self.to_user_page()
        self.get_user_profile()
        self.get_all_post_detail()
        logging.info(f'爬取【{self.user_name}】完毕，保存数据成功')
        self.save_posts_to_csv(self.res)
        return self.sum_data()


def main(task_list: List[str]):
    all_res = []
    for t in task_list:
        xhs = RedBook(user_name=t)
        all_res.append(xhs.run())
    save_dict_list_to_csv(all_res, filename='red_book_data.csv')


if __name__ == '__main__':

    # 传一个待搜索列表给 main() 函数执行即可
    target_list = ['迪丽热巴', '杨幂', '孙俪']
    main(target_list)

from DrissionPage import ChromiumPage

class WeiboCrawler:
    def __init__(self):
        self.page = ChromiumPage()

    def search_weibo(self, keyword):
        self.page.get('http://s.weibo.com/weibo?q=' + keyword)
        self.page.wait.load_start()
        self.page.wait(3)

    def get_names(self):
        elements = self.page.eles('css:.name')
        names = [element.text for element in elements]
        return names

if __name__ == "__main__":
    weibo = WeiboCrawler()
    weibo.search_weibo('享界S9')
    names = weibo.get_names()
    for name in names:
        print(name)



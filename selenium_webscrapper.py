from selenium import webdriver
import time
import os
import requests
from PIL import Image


def progress_bar(iteration, total, status):  # just a little feature
    bar_length = 60
    percentage = iteration/total
    filled_bar = round(percentage*bar_length)
    print('\r', '[' + '▓' * filled_bar + '░' * (bar_length-filled_bar) + ']', '[{:>7.2%}] '.format(percentage) + status, end='')


def scrap_images(queries, starting_index, master_query='кошка'):
    start = time.time()
    browser = webdriver.Chrome(r"C:\Users\peter\PycharmProjects\ML_BC\chromedriver.exe")
    total_num = 0
    starting_index = starting_index

    for query in queries.keys():
        total_num += queries[query]

    for query in queries.keys():
        y = 0
        if query == master_query:
            y = 1
        num_of_images = queries[query]
        browser.get(f'https://yandex.ru/images/search?text={query}&iorient=square')
        browser.maximize_window()
        for i in range(int(num_of_images/40)+1):  # scrolling page down
            browser.execute_script('scrollBy(' + str(i * 1000) + ',+1000);')
            time.sleep(3)

        container_div = browser.find_element_by_class_name('serp-list')  # class names may change
        images = container_div.find_elements_by_tag_name('img')

        try:
            os.mkdir(r'E:\Peter\parsed_images')
        except FileExistsError:
            pass

        for i in range(starting_index, num_of_images+starting_index):
            progress_bar(i-starting_index, total_num, 'working...')
            url = images[i-starting_index].get_attribute('src')
            try:
                if url is not None:
                    url = str(url)
                    response = requests.get(url)
                    if response.status_code == 200:
                        image_path = r'E:\Peter\parsed_images\\%i.jpg' % i
                        file = open(image_path, 'wb')
                        file.write(response.content)
                        file.close()
                        image = Image.open(image_path)
                        resized_image = image.resize((64, 64), Image.ANTIALIAS)
                        resized_image.save(r'E:\Peter\parsed_images\\%i_%i.jpg' % (i, y))
                        os.remove(image_path)
            except TypeError:
                print('failed')
            except OSError:
                pass
        starting_index += num_of_images
    browser.close()
    progress_bar(total_num, total_num, 'done! \n')
    print('got ' + str(total_num) + ' images in ' + str(round(((time.time() - start) * 1000), 2)) + ' ms')


if __name__ == '__main__':  # could (better to) pass queries via array
    scrap_images({'собака': 1300}, 1300, 'кошка')

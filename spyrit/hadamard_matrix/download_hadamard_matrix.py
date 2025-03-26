import numpy as np
import requests

def read_text_file_from_url(url):
    response = requests.get(url)
    content = response.text
    return content

def download_from_sloane():
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    # Set up the WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Open the website
    driver.get("http://neilsloane.com/hadamard/")

    # Find all links to Hadamard matrices
    links = driver.find_elements(By.XPATH, "//a[contains(@href, 'had.')]")

    # Extract the URLs
    hadamard_urls = set([link.get_attribute('href') for link in links])

    # Print the URLs
    for url in hadamard_urls:
        print(url)
        # Read the text file from the URL
        file_content = read_text_file_from_url(url)
        # Split the content into lines
        lines = file_content.splitlines()

        # Print the content of the file
        if '+' in file_content or '0' in file_content or '-1' in file_content:
            if len(lines) > 1:
                size = len(lines[1])
            else:
                size = len(lines[0])
            array = []
            for line in lines:
                if len(line) == size:
                    line = line.replace("-1", "0")
                    tmp = []
                    for e in line:
                        if e == '+' or e == '1':
                            tmp += [1]
                        elif e == '-' or e == '0':
                            tmp += [0]
                        elif e == " ":
                            pass
                        else:
                            print("Error during reading of " + url)
                    array += [tmp]
            np_array = np.array(array, dtype=bool)
            name = url.split('/')[-1][:-4]
            np.savez_compressed(name + '.npz', np_array)
        else:
            print("no ok for " + url)
        #print(file_content)

    # Close the WebDriver
    driver.quit()

if __name__ == "__main__":
    download_from_sloane()


import numpy as np
import requests
import os
import glob
import importlib.util
import tqdm


def download_from_girder():
    """
    Download Hadamard matrices from the Girder repository into hadamard_matrix folder.
    """

    hadamard_matrix_path = os.path.dirname(__file__)
    if os.path.isfile(
        os.path.join(hadamard_matrix_path, "had.236.sage.cooper-wallis.npz")
    ):
        return
    print("Downloading Hadamard matrices (>2300) from Girder repository...")
    print(
        "The matrices were downloaded from http://neilsloane.com/hadamard/ Sloane et al."
    )
    import girder_client

    gc = girder_client.GirderClient(
        apiUrl="https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
    )

    collection_id = "66796d3cbaa5a90007058946"
    folder_id = "6800c6891240141f6aa53845"
    limit = 50  # Number of items to retrieve per request
    offset = 0  # Starting point
    pbar = tqdm.tqdm(total=0)

    while True:
        items = gc.get(
            "item",
            parameters={
                "parentType": "collection",
                "parentId": collection_id,
                "folderId": folder_id,
                "limit": limit,
                "offset": offset,
            },
        )
        if not items:
            break
        pbar.total += len(items)
        pbar.refresh()
        for item in items:
            files = gc.get(f'item/{item["_id"]}/files')
            for file in files:
                pbar.update(1)
                gc.downloadFile(
                    file["_id"], os.path.join(hadamard_matrix_path, file["name"])
                )
        offset += limit
    pbar.close()


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
    hadamard_urls = set([link.get_attribute("href") for link in links])

    # Print the URLs
    for url in hadamard_urls:
        print(url)
        # Read the text file from the URL
        file_content = read_text_file_from_url(url)
        # Split the content into lines
        lines = file_content.splitlines()

        # Print the content of the file
        if "+" in file_content or "0" in file_content or "-1" in file_content:
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
                        if e == "+" or e == "1":
                            tmp += [1]
                        elif e == "-" or e == "0":
                            tmp += [0]
                        elif e == " ":
                            pass
                        else:
                            print("Error during reading of " + url)
                    array += [tmp]
            np_array = np.array(array, dtype=bool)

            name = url.split("/")[-1][:-4]
            order = int(name.split(".")[1])

            # Check if the file already exists
            files = glob.glob("had." + str(order) + "*.npz")
            already_saved = False
            for file in files:
                b = np.load(file)
                if np.all(np_array == b):
                    already_saved = True
                if already_saved:
                    break

            if not already_saved:
                np.savez_compressed(name + ".npz", np_array)
        else:
            print("no ok for " + url)
        # print(file_content)

    # Close the WebDriver
    driver.quit()


if __name__ == "__main__":
    download_from_sloane()

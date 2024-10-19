const scanAllButton = document.getElementById("scanAllButton");
const scanTextButton = document.getElementById("scanTextButton");
const scanImagesButton = document.getElementById("scanImagesButton");
const scanVideosButton = document.getElementById("scanVideosButton");
const scanButtons = [scanAllButton, scanTextButton, scanImagesButton, scanVideosButton];

scanAllButton.addEventListener("click", scanAll);
scanTextButton.addEventListener("click", scanText);
scanImagesButton.addEventListener("click", scanImages);
scanVideosButton.addEventListener("click", scanVideos);

const mainPage = document.getElementById("main-page");


function scanAll() {
    disableButtons();
    scanText();
}

function scanText() {
    disableButtons()
    chrome.tabs.query({active: true, currentWindow: true})
        .then(tabs => {
            chrome.tabs.sendMessage(tabs[0].id, {action: "scanText"})
        })
        .then(result => {
            console.log("Scan Complete");
            enableButtons();
        });
}

//------------------- Images -------------------

const imagesPage = document.getElementById("images-page");
const imagesList = document.getElementById("images-list");
const imagesBackButton = document.getElementById("images-back-button");
const imagesPrevButton = document.getElementById("images-prev-button");
const imagesNextButton = document.getElementById("images-next-button");
let currentImagePage = 1; // Track the current page
const imagesPerPage = 4; // Number of images to show per page

imagesBackButton.addEventListener("click", () => {
    imagesPage.classList.toggle("hidden");
    mainPage.classList.toggle("hidden");
});

imagesPrevButton.addEventListener("click", showPrevImages);
imagesNextButton.addEventListener("click", showNextImages);



function scanImages() {
    disableButtons();
    // Query the current tab to send a message to the content script
    chrome.tabs.query({ active: true, currentWindow: true })
        .then(tabs => {
            chrome.tabs.sendMessage(tabs[0].id, { action: "scanImages" }, (response) => {
                if (response && response.images) {
                    // Hide main content and show images container
                    
                    imagesPage.classList.toggle("hidden");
                    mainPage.classList.toggle("hidden");

                    // Clear existing images from the list
                    imagesList.innerHTML = '';

                    // Add each image and a scan button
                    response.images.forEach((imageUrl, index) => {
                        const img = document.createElement("img");
                        img.src = imageUrl;
                        img.style.maxWidth = "150px";
                        img.style.maxHeight = "100px";
                        img.className = "flex-auto"; // Tailwind classes for styling

                        const scanButton = document.createElement("button");
                        scanButton.textContent = "Scan";
                        scanButton.className = "button flex-none";
                        scanButton.addEventListener("click", () => {
                            scanImage(img.src);
                        });
                        
                        // Create images, but make them hidden for now
                        const imageScanPair = document.createElement("div");
                        imageScanPair.className = "flex place-items-center justify-center gap-4"; // Flexbox layout

                        imageScanPair.appendChild(img);
                        imageScanPair.appendChild(scanButton);
                        imagesList.appendChild(imageScanPair);
                    });

                    // Show only imagesPerPage images on the currentPage
                    showImagesPerPage();
                }

                enableButtons();
            });
        });
}

function showImagesPerPage() {
    const images = imagesList.children;
    for (let i = 0; i < images.length; i++) {
        if (i >= (currentImagePage - 1) * imagesPerPage && i < currentImagePage * imagesPerPage) {
            images[i].classList.remove("hidden");
        } else {
            images[i].classList.add("hidden");
        }
    }
}

function showNextImages() {
    if (currentImagePage * imagesPerPage < imagesList.children.length) {
        currentImagePage++;
        showImagesPerPage();
    }
}

function showPrevImages() {
    if (currentImagePage > 1) {
        currentImagePage--;
        showImagesPerPage();
    }
}

function scanImage(imageUrl) {
    disableButtons();
    chrome.tabs.query({active: true, currentWindow: true})
        .then(tabs => {
            chrome.tabs.sendMessage(tabs[0].id, {action: "scanImage", imgUrl: imageUrl})
        })
        .then(result => {
            // TODO: Display result or something
            console.log(result.data);
            enableButtons();
        });
}

// ------------------------------------------------

function scanVideos() {
    disableButtons();
    // TODO
    enableButtons();
}

function disableButtons() {
    scanButtons.forEach(button => {
        button.disabled = true;
    });
}

function enableButtons() {
    scanButtons.forEach(button => {
        button.disabled = false;
    });
}

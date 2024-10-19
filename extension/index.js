const scanAllButton = document.getElementById("scanAllButton");
const scanTextButton = document.getElementById("scanTextButton");
const scanImagesButton = document.getElementById("scanImagesButton");
const scanVideosButton = document.getElementById("scanVideosButton");
const scanButtons = [scanAllButton, scanTextButton, scanImagesButton, scanVideosButton];

scanAllButton.addEventListener("click", scanAll);
scanTextButton.addEventListener("click", scanText);
scanImagesButton.addEventListener("click", scanImages);
scanVideosButton.addEventListener("click", scanVideos);

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

function scanImages() {
    disableButtons();
    // TODO
    enableButtons();
}

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

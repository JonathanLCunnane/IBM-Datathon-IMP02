const scanButton = document.getElementById('scanButton');

scanButton.addEventListener('click', scanPage);

const resultsDiv = document.getElementById('result');

function scanPage() {
    scanButton.disabled = true;
    scanButton.innerHTML = 'Loading...';
    chrome.tabs.query({active: true, currentWindow: true})
    .then(tabs => {
        chrome.tabs.sendMessage(tabs[0].id, {action: 'scanPage'})
    })
    .then(result => {
        console.log(result);
        resultsDiv.classList.toggle('hidden');
        scanButton.innerHTML = 'Scan Complete';
    });
}


var modal = document.getElementById("myModal");
var btn = document.getElementById("openModal");
var span = document.getElementsByClassName("close")[0];

function showUploadNotification(event) {
    event.preventDefault();
    alert("Please return to the homepage to upload files...");
    window.location.href = '../../';
}

function showUploadNotifications(event) {
    event.preventDefault();
    alert("Please return to the homepage to upload files...");
    window.location.href = '../../IRW';
}

btn.onclick = function() {
    modal.style.display = "block";
}

span.onclick = function() {
    modal.style.display = "none";
}

window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
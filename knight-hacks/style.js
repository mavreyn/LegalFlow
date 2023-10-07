let slideIndex = 1; // start on first slide
showSlide(slideIndex);

function moveSlide(n) {
  showSlide((slideIndex += n));
}

function showSlide(n) {
  let i;
  let slides = document.getElementsByClassName("slide");
  if (n > slides.length) {
    slideIndex = 1;
  }
  if (n < 1) {
    slideIndex = slides.length;
  }
  for (i = 0; i < slides.length; i++) {
    slides[i].classList.remove("active");
  }
  slides[slideIndex - 1].classList.add("active");
}

document.addEventListener("DOMContentLoaded", function () {
  let prevButton = document.querySelector(".prev");
  let nextButton = document.querySelector(".next");

  prevButton.addEventListener("click", function () {
    moveSlide(-1);
  });

  nextButton.addEventListener("click", function () {
    moveSlide(1);
  });
});

let slideInterval = setInterval(function () {
  moveSlide(1);
}, 5000);

document.getElementById("questionForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = document.getElementById("question").value;
  const response = await fetch("/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  const data = await response.json();
  document.getElementById("answer").innerText = data.answer;
});

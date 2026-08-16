    const form = document.getElementById('login-form');
    const input = document.getElementById('passcode');
    const submitBtn = document.getElementById('submit-btn');
    const message = document.getElementById('message');

    function setMessage(text){
      message.textContent = text || '';
      message.classList.toggle('visible', !!text);
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const passcode = input.value || '';
      submitBtn.disabled = true;
      setMessage('');
      try{
        const response = await fetch('/api/lan/auth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ passcode }),
        });
        const data = await response.json().catch(() => ({}));
        if(!response.ok){
          setMessage(data && data.error ? String(data.error) : 'incorrect passcode');
          input.select();
          return;
        }
        window.location.replace('/');
      }catch(_error){
        setMessage('could not verify passcode');
      }finally{
        submitBtn.disabled = false;
      }
    });

    setTimeout(() => input.focus(), 60);

(() => {
      const data = window.GUQIN_GALLERY_DATA;
      if (!data) { document.body.textContent = '试听数据未找到。请保留 index.html、data.js 和 audio 文件夹。'; return; }
      const key = 'guqin-listening-feedback-v1';
      let feedback = {};
      try { feedback = JSON.parse(localStorage.getItem(key) || '{}') || {}; } catch { feedback = {}; }
      const issues = ['急促音','机械纯音','滑到突兀半音','短音串','底噪重','疑似其他乐器'];
      const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2,'0')}`;
      const save = () => { try { localStorage.setItem(key, JSON.stringify(feedback)); } catch {} };
      const state = id => (feedback[id] ||= {vote:'', issues:[], note:''});
      const player = (id, variant='') => {
        const sample = data.samples[id];
        return `<div class="player ${variant}" data-sample-id="${id}"><div class="player-title"><span class="dot"></span>${esc(sample.title)}<span class="duration">${formatTime(sample.duration)}</span></div><div class="wave" role="slider" tabindex="0" aria-label="跳转 ${esc(sample.title)} 播放进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" data-wave="${id}">${sample.waveform.map(height => `<span style="--h:${height}%"></span>`).join('')}</div><audio controls preload="none" src="${esc(sample.src)}" aria-label="播放 ${esc(sample.title)}"></audio></div>`;
      };
      document.getElementById('stats').innerHTML = `<span class="stat">${data.pairs.length} 组同条件对照</span><span class="stat">${Object.keys(data.samples).length} 段音频</span><span class="stat">本地离线试听</span>`;
      document.getElementById('pair-list').innerHTML = data.pairs.map((pair,index) => {
        const saved = state(pair.id);
        return `<article class="pair" data-pair="${pair.id}"><div class="pair-head"><div><span class="pair-index">${String(index+1).padStart(2,'0')}</span><h3>${esc(pair.title)}</h3><p class="pair-desc">${esc(pair.description)}</p></div><span class="badge">A / B</span></div><div class="players">${player(pair.old)}${player(pair.new,'new')}</div><div class="feedback"><div class="feedback-row"><span class="feedback-label">更喜欢哪版</span>${[['old',data.samples[pair.old].title],['new',data.samples[pair.new].title],['equal','差不多'],['neither','都不满意']].map(([value,label]) => `<button type="button" class="choice ${saved.vote===value?'selected':''}" data-vote="${value}" aria-pressed="${saved.vote===value}">${label}</button>`).join('')}</div><div class="issue-row"><span class="feedback-label">听到的问题</span>${issues.map(issue => `<button type="button" class="chip ${(saved.issues||[]).includes(issue)?'selected':''}" data-issue="${esc(issue)}" aria-pressed="${(saved.issues||[]).includes(issue)}">${esc(issue)}</button>`).join('')}</div>${pair.note?`<p class="known">已知反馈：${esc(pair.note)}</p>`:''}<div class="notes"><textarea data-note="${pair.id}" placeholder="记录时间和具体听感，例如 0:14 急促、像古筝……" aria-label="${esc(pair.title)} 试听笔记">${esc(saved.note)}</textarea><button type="button" class="mark" data-mark="${pair.id}" title="从正在播放的样本记录时间">记时间</button></div></div></article>`;
      }).join('');
      const archiveCard = entry => { const s=data.samples[entry.sample]; return `<article class="archive-item" data-search="${esc((s.title+' '+entry.group).toLowerCase())}"><span class="group">${esc(entry.group)}</span><h3>${esc(s.title)}</h3>${player(entry.sample)}<textarea data-note="${entry.sample}" placeholder="这段听起来怎么样？" aria-label="${esc(s.title)} 试听笔记">${esc(state(entry.sample).note)}</textarea></article>`; };
      document.getElementById('archive-grid').innerHTML = data.extras.map(archiveCard).join('');
      document.getElementById('reference-grid').innerHTML = data.references.map(archiveCard).join('');
      document.querySelectorAll('audio').forEach(audio => {
        audio.addEventListener('play', () => document.querySelectorAll('audio').forEach(other => { if (other !== audio) other.pause(); }));
        audio.addEventListener('timeupdate', () => { const wave=audio.parentElement.querySelector('.wave'); if (audio.duration) { const percent=Math.min(100,100*audio.currentTime/audio.duration); wave.style.setProperty('--progress',percent+'%'); wave.setAttribute('aria-valuenow',String(Math.round(percent))); } });
      });
      document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
        const name=button.dataset.view;
        document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-selected',String(b===button)));
        document.querySelectorAll('.view').forEach(view => view.hidden = view.id !== 'view-'+name);
        document.querySelectorAll('audio').forEach(audio => audio.pause());
      }));
      document.addEventListener('click', event => {
        const vote=event.target.closest('[data-vote]');
        if (vote) { const card=vote.closest('[data-pair]'); const s=state(card.dataset.pair); s.vote=vote.dataset.vote; card.querySelectorAll('[data-vote]').forEach(b => { const selected=b===vote; b.classList.toggle('selected',selected); b.setAttribute('aria-pressed',String(selected)); }); save(); return; }
        const issue=event.target.closest('[data-issue]');
        if (issue) { const s=state(issue.closest('[data-pair]').dataset.pair); s.issues ||= []; const value=issue.dataset.issue; s.issues=s.issues.includes(value)?s.issues.filter(x=>x!==value):[...s.issues,value]; const selected=s.issues.includes(value); issue.classList.toggle('selected',selected); issue.setAttribute('aria-pressed',String(selected)); save(); return; }
        const mark=event.target.closest('[data-mark]');
        if (mark) { const card=mark.closest('[data-pair]'); const active=[...card.querySelectorAll('audio')].find(a=>!a.paused)||card.querySelector('audio'); const which=data.samples[active.parentElement.dataset.sampleId].title; const field=card.querySelector('textarea'); field.value += `${field.value?'\n':''}${which} ${formatTime(active.currentTime)}：`; field.focus(); state(card.dataset.pair).note=field.value; save(); return; }
        const wave=event.target.closest('[data-wave]');
        if (wave) { const audio=wave.parentElement.querySelector('audio'); if (audio.duration) audio.currentTime = Math.max(0,Math.min(1,(event.clientX-wave.getBoundingClientRect().left)/wave.getBoundingClientRect().width))*audio.duration; }
      });
      document.addEventListener('keydown', event => { if ((event.key==='Enter'||event.key===' ') && event.target.matches('[data-wave]')) { event.preventDefault(); const audio=event.target.parentElement.querySelector('audio'); if (audio.duration) audio.currentTime=Math.min(audio.duration,audio.currentTime+5); } });
      document.querySelectorAll('[data-note]').forEach(field => field.addEventListener('input', () => { state(field.dataset.note).note=field.value; save(); }));
      document.getElementById('archive-search').addEventListener('input', event => { const query=event.target.value.trim().toLowerCase(); let shown=0; document.querySelectorAll('#archive-grid .archive-item').forEach(card=> { card.hidden=!card.dataset.search.includes(query); if (!card.hidden) shown++; }); document.getElementById('archive-empty').style.display=shown?'none':'block'; });
      document.getElementById('export').addEventListener('click', () => { const payload={exportedAt:new Date().toISOString(),page:'听琴台',feedback}; const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'})); const link=document.createElement('a'); link.href=url; link.download='古琴试听反馈.json'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000); });
    })();

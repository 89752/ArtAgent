import { useChatStore } from "../store/chatStore";

export function Welcome() {
  const cards = useChatStore((s) => s.cards);
  const send = useChatStore((s) => s.send);

  return (
    <section id="welcome" className="welcome">
      <div className="w-inner">
        <img className="w-mark" src="/static/emblem.svg" alt="天使徽标" />
        <h1 className="w-title">
          <span className="flr flr-l" aria-hidden="true" />
          <span>西方艺术智能助手</span>
          <span className="flr flr-r" aria-hidden="true" />
        </h1>
        <div className="w-divider" aria-hidden="true" />
        <p className="w-greeting">您好，我是您的西方艺术智能助手。</p>
        <p className="w-hint">
          向我提问关于艺术作品、艺术家、风格流派与历史背景的一切。
        </p>
        <div id="cards" className="cards">
          {cards.map((c, i) => (
            <button
              key={i}
              type="button"
              className="scene-card"
              onClick={() => void send(c.query)}
            >
              {c.thumb && (
                <img
                  className="sc-thumb"
                  src={c.thumb}
                  alt=""
                  loading="lazy"
                />
              )}
              <div className="sc-body">
                <div className="sc-title">{c.text}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

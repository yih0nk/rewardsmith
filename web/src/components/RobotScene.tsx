export default function RobotScene({
  playing,
  round,
}: {
  playing: boolean;
  round: number;
}) {
  return (
    <div className={`scene ${playing ? "" : "paused"}`}>
      <span className="scene-label">
        <i /> {round ? "POLICY ROLLOUT" : "PREPARING POLICY"}
      </span>
      <div className="floor" />
      <svg
        viewBox="0 0 720 330"
        role="img"
        aria-label="Illustrative animated quadruped on a simulation grid"
      >
        <defs>
          <linearGradient id="body" x2="0.8" y2="1">
            <stop stopColor="#f8faf8" />
            <stop offset="1" stopColor="#9daaa4" />
          </linearGradient>
        </defs>
        <ellipse
          cx="365"
          cy="267"
          rx="115"
          ry="17"
          fill="#324a3a"
          opacity=".12"
        />
        <g className="robot">
          <g
            stroke="#414e48"
            strokeWidth="13"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          >
            <path className="leg back" d="M310 172 L290 218 L315 256" />
            <path className="leg front" d="M406 172 L430 212 L413 253" />
          </g>
          <path
            d="M278 141 L388 125 L445 151 L335 174 Z"
            fill="#e3eae5"
            stroke="#697b70"
          />
          <path d="M278 141 L335 165 L335 194 L278 168 Z" fill="#8c9d92" />
          <path
            d="M335 165 L445 143 L445 175 L335 195 Z"
            fill="url(#body)"
            stroke="#819489"
          />
          <path d="M290 139 L382 126 L419 141 L327 157 Z" fill="#f7faf7" />
          <path d="M315 133 L365 126 L387 136 L336 145 Z" fill="#f3bd65" />
          <rect x="426" y="140" width="35" height="27" rx="8" fill="#344c40" />
          <circle cx="451" cy="151" r="4" fill="#b9eec8" />
          <g
            stroke="#35463d"
            strokeWidth="14"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          >
            <path className="leg front" d="M301 171 L326 216 L299 259" />
            <path className="leg back" d="M413 182 L388 221 L414 263" />
          </g>
          <g fill="#e4eae5" stroke="#718579" strokeWidth="3">
            <circle cx="301" cy="171" r="12" />
            <circle cx="413" cy="182" r="12" />
          </g>
        </g>
        <g transform="translate(72 255)" strokeWidth="2">
          <path d="M0 0 L35 0" stroke="#ad6c59" />
          <path d="M0 0 L0 -35" stroke="#6e9177" />
          <path d="M0 0 L-18 15" stroke="#7a91a4" />
        </g>
        <text x="114" y="259" fill="#9b7a6d" fontSize="10">
          X
        </text>
        <text x="69" y="212" fill="#6e9177" fontSize="10">
          Z
        </text>
      </svg>
      <span className="scene-bottom">
        QUADRUPED / {round ? `ITERATION 0${round}` : "STANDBY"}
      </span>
      <span className="scene-coords">x −0.64 &nbsp; y 0.02 &nbsp; z 0.32</span>
    </div>
  );
}

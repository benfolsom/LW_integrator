<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 440" role="img" aria-labelledby="title desc">
  <title id="title">Full retarded solver versus pseudo-grid reduced model</title>
  <desc id="desc">Comparison of all-to-all retarded interactions and pseudo-grid active/passive reduced interactions.</desc>
  <defs>
    <marker id="arrow-full" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,4 L0,8 Z" fill="#2f6fb0" opacity="0.7"/>
    </marker>
    <marker id="arrow-reduced" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,4 L0,8 Z" fill="#7a3fb0" opacity="0.85"/>
    </marker>
    <marker id="arrow-passive" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,4 L0,8 Z" fill="#777" opacity="0.75"/>
    </marker>
    <style>
      .panel { fill: #f8fafc; stroke: #c9d2dc; stroke-width: 1.4; rx: 14; }
      .title { font-family: sans-serif; font-size: 21px; font-weight: 700; fill: #172033; }
      .subtitle { font-family: sans-serif; font-size: 13px; fill: #344054; }
      .label { font-family: sans-serif; font-size: 12px; fill: #263238; }
      .small { font-family: sans-serif; font-size: 11px; fill: #475467; }
      .full-edge { stroke: #2f6fb0; stroke-width: 1.15; opacity: 0.34; marker-end: url(#arrow-full); }
      .reduced-edge { stroke: #7a3fb0; stroke-width: 2.2; opacity: 0.72; marker-end: url(#arrow-reduced); }
      .passive-edge { stroke: #777; stroke-width: 1.4; stroke-dasharray: 4 4; opacity: 0.66; marker-end: url(#arrow-passive); }
      .active { fill: #7a3fb0; stroke: #32105f; stroke-width: 1.5; }
      .passive { fill: #ffffff; stroke: #7a3fb0; stroke-width: 1.4; }
      .full-rider { fill: #1f77b4; stroke: #0f3f63; stroke-width: 1.4; }
      .full-driver { fill: #ff9f1c; stroke: #915500; stroke-width: 1.4; }
      .driver-active { fill: #ff9f1c; stroke: #915500; stroke-width: 1.5; }
      .driver-passive { fill: #ffffff; stroke: #ff9f1c; stroke-width: 1.4; }
      .badge { fill: #eef4ff; stroke: #98b7e0; stroke-width: 1; rx: 8; }
      .badge2 { fill: #f4efff; stroke: #bfa4e6; stroke-width: 1; rx: 8; }
    </style>
  </defs>

  <rect x="18" y="24" width="454" height="392" class="panel"/>
  <rect x="508" y="24" width="454" height="392" class="panel"/>

  <text x="44" y="58" class="title">Nominal fully retarded solve</text>
  <text x="44" y="80" class="subtitle">Every observer samples every source history on each step</text>
  <text x="44" y="98" class="small">All pairwise retarded interactions are evaluated directly.</text>

  <text x="535" y="58" class="title">Pseudo-grid reduced solve</text>
  <text x="535" y="80" class="subtitle">Only active anchors solve; passives aggregate and reconstruct</text>

  <!-- Full solver points -->
  <g id="full-rider">
    <circle cx="90" cy="150" r="8" class="full-rider"/>
    <circle cx="145" cy="115" r="8" class="full-rider"/>
    <circle cx="155" cy="185" r="8" class="full-rider"/>
    <circle cx="210" cy="145" r="8" class="full-rider"/>
    <circle cx="105" cy="225" r="8" class="full-rider"/>
    <circle cx="225" cy="235" r="8" class="full-rider"/>
  </g>
  <g id="full-driver">
    <circle cx="302" cy="130" r="8" class="full-driver"/>
    <circle cx="360" cy="110" r="8" class="full-driver"/>
    <circle cx="390" cy="170" r="8" class="full-driver"/>
    <circle cx="322" cy="215" r="8" class="full-driver"/>
    <circle cx="405" cy="245" r="8" class="full-driver"/>
    <circle cx="350" cy="285" r="8" class="full-driver"/>
  </g>

  <!-- Full all-to-all interaction fan -->
  <g>
    <line x1="90" y1="150" x2="302" y2="130" class="full-edge"/>
    <line x1="90" y1="150" x2="360" y2="110" class="full-edge"/>
    <line x1="90" y1="150" x2="390" y2="170" class="full-edge"/>
    <line x1="90" y1="150" x2="322" y2="215" class="full-edge"/>
    <line x1="90" y1="150" x2="405" y2="245" class="full-edge"/>
    <line x1="90" y1="150" x2="350" y2="285" class="full-edge"/>

    <line x1="145" y1="115" x2="302" y2="130" class="full-edge"/>
    <line x1="145" y1="115" x2="360" y2="110" class="full-edge"/>
    <line x1="145" y1="115" x2="390" y2="170" class="full-edge"/>
    <line x1="145" y1="115" x2="322" y2="215" class="full-edge"/>
    <line x1="145" y1="115" x2="405" y2="245" class="full-edge"/>
    <line x1="145" y1="115" x2="350" y2="285" class="full-edge"/>

    <line x1="155" y1="185" x2="302" y2="130" class="full-edge"/>
    <line x1="155" y1="185" x2="360" y2="110" class="full-edge"/>
    <line x1="155" y1="185" x2="390" y2="170" class="full-edge"/>
    <line x1="155" y1="185" x2="322" y2="215" class="full-edge"/>
    <line x1="155" y1="185" x2="405" y2="245" class="full-edge"/>
    <line x1="155" y1="185" x2="350" y2="285" class="full-edge"/>

    <line x1="210" y1="145" x2="302" y2="130" class="full-edge"/>
    <line x1="210" y1="145" x2="360" y2="110" class="full-edge"/>
    <line x1="210" y1="145" x2="390" y2="170" class="full-edge"/>
    <line x1="210" y1="145" x2="322" y2="215" class="full-edge"/>
    <line x1="210" y1="145" x2="405" y2="245" class="full-edge"/>
    <line x1="210" y1="145" x2="350" y2="285" class="full-edge"/>

    <line x1="105" y1="225" x2="302" y2="130" class="full-edge"/>
    <line x1="105" y1="225" x2="360" y2="110" class="full-edge"/>
    <line x1="105" y1="225" x2="390" y2="170" class="full-edge"/>
    <line x1="105" y1="225" x2="322" y2="215" class="full-edge"/>
    <line x1="105" y1="225" x2="405" y2="245" class="full-edge"/>
    <line x1="105" y1="225" x2="350" y2="285" class="full-edge"/>

    <line x1="225" y1="235" x2="302" y2="130" class="full-edge"/>
    <line x1="225" y1="235" x2="360" y2="110" class="full-edge"/>
    <line x1="225" y1="235" x2="390" y2="170" class="full-edge"/>
    <line x1="225" y1="235" x2="322" y2="215" class="full-edge"/>
    <line x1="225" y1="235" x2="405" y2="245" class="full-edge"/>
    <line x1="225" y1="235" x2="350" y2="285" class="full-edge"/>
  </g>

  <rect x="58" y="325" width="355" height="58" class="badge"/>
  <text x="75" y="344" class="label">Cost per cross-bunch step: all observer-source pairs</text>
  <text x="75" y="361" class="small">Example shown: 6 rider observers × 6 driver sources</text>
  <text x="75" y="375" class="small">= 36 retarded samples</text>

  <!-- Reduced solver points -->
  <g id="reduced-rider">
    <circle cx="590" cy="158" r="8" class="active"/>
    <circle cx="642" cy="128" r="7" class="passive"/>
    <circle cx="655" cy="195" r="8" class="active"/>
    <circle cx="710" cy="150" r="7" class="passive"/>
    <circle cx="608" cy="235" r="7" class="passive"/>
    <circle cx="725" cy="245" r="8" class="active"/>
  </g>
  <g id="reduced-driver">
    <circle cx="790" cy="138" r="8" class="driver-active"/>
    <circle cx="848" cy="118" r="7" class="driver-passive"/>
    <circle cx="888" cy="182" r="8" class="driver-active"/>
    <circle cx="820" cy="228" r="7" class="driver-passive"/>
    <circle cx="900" cy="252" r="7" class="driver-passive"/>
    <circle cx="848" cy="292" r="8" class="driver-active"/>
  </g>

  <!-- Passive aggregation/reconstruction within bunches -->
  <g>
    <line x1="642" y1="128" x2="590" y2="158" class="passive-edge"/>
    <line x1="710" y1="150" x2="655" y2="195" class="passive-edge"/>
    <line x1="608" y1="235" x2="590" y2="158" class="passive-edge"/>
    <line x1="608" y1="235" x2="655" y2="195" class="passive-edge"/>
    <line x1="848" y1="118" x2="790" y2="138" class="passive-edge"/>
    <line x1="820" y1="228" x2="790" y2="138" class="passive-edge"/>
    <line x1="820" y1="228" x2="848" y2="292" class="passive-edge"/>
    <line x1="900" y1="252" x2="888" y2="182" class="passive-edge"/>
    <line x1="900" y1="252" x2="848" y2="292" class="passive-edge"/>
  </g>

  <!-- Active-to-active reduced interactions -->
  <g>
    <line x1="590" y1="158" x2="790" y2="138" class="reduced-edge"/>
    <line x1="590" y1="158" x2="888" y2="182" class="reduced-edge"/>
    <line x1="590" y1="158" x2="848" y2="292" class="reduced-edge"/>
    <line x1="655" y1="195" x2="790" y2="138" class="reduced-edge"/>
    <line x1="655" y1="195" x2="888" y2="182" class="reduced-edge"/>
    <line x1="655" y1="195" x2="848" y2="292" class="reduced-edge"/>
    <line x1="725" y1="245" x2="790" y2="138" class="reduced-edge"/>
    <line x1="725" y1="245" x2="888" y2="182" class="reduced-edge"/>
    <line x1="725" y1="245" x2="848" y2="292" class="reduced-edge"/>
  </g>

  <text x="552" y="148" class="small" text-anchor="middle">active anchors</text>
  <text x="632" y="108" class="small">passive rider nodes</text>
  <text x="900" y="160" class="small" text-anchor="middle">active sources</text>

  <rect x="548" y="325" width="380" height="58" class="badge2"/>
  <text x="565" y="344" class="label">Cost per cross-bunch step: active observer-source pairs</text>
  <text x="565" y="361" class="small">Example shown: 3 active rider × 3 active driver = 9 retarded samples</text>
  <text x="565" y="375" class="small">Each active source stands in for nearby passive-source charge.</text>

  <g>
    <circle cx="68" cy="398" r="6" class="full-rider"/>
    <text x="82" y="402" class="small">rider particle</text>
    <circle cx="170" cy="398" r="6" class="full-driver"/>
    <text x="184" y="402" class="small">driver particle</text>
    <circle cx="557" cy="398" r="6" class="active"/>
    <text x="571" y="402" class="small">active</text>
    <circle cx="630" cy="398" r="6" class="passive"/>
    <text x="644" y="402" class="small">passive</text>
    <line x1="700" y1="398" x2="742" y2="398" class="reduced-edge"/>
    <text x="752" y="394" class="small">retarded</text>
    <text x="752" y="408" class="small">active solve</text>
    <line x1="838" y1="398" x2="874" y2="398" class="passive-edge"/>
    <text x="884" y="394" class="small">local</text>
    <text x="884" y="408" class="small">weighting</text>
  </g>
</svg>

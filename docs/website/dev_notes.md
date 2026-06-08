<style>
/* =========================
   LAYOUT
========================= */

.md-sidebar--primary,
.md-sidebar--secondary {
  display: none !important;
}

.md-grid {
  max-width: 1400px !important;
}

.md-main__inner {
  display: flex !important;
  justify-content: center !important;
}

.md-content {
  margin: 0 auto !important;
}

.md-content__inner {
  max-width: 1000px !important;
  margin: 0 auto !important;
  padding-top: 0 !important;
}

.md-content__inner:before {
  display: none !important;
}

/* =========================
   ROADMAP TIMELINE
========================= */

.roadmap-section {
  border: 1px solid #e5e7eb;
  border-radius: 20px;
  padding: 2rem 2rem 2rem 2rem;
  background: white;
  margin-bottom: 2rem;
}

.roadmap-title-main {
  margin: 0;
  color: #185FA5;
  font-size: 32px;
  font-weight: 600;
  letter-spacing: -0.03em;
  margin-bottom: 0.75rem;
}

.roadmap-subtitle {
  color: #6b7280;
  font-size: 14px;
  line-height: 1.7;
  margin-top: 0.7rem;
  margin-bottom: 1rem;
}

/* Timeline container */

.timeline {
  position: relative;
  padding: 1rem 0;
}

/* Main horizontal line */

.timeline-line {
  position: absolute;
  top: 50%;
  left: 0;
  width: 100%;
  height: 3px;
  background: #dbe7f3;
  transform: translateY(-50%);
}

/* Grid */

.timeline-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1.5rem;
  position: relative;
  z-index: 2;
}

/* Timeline item */

.timeline-item {
  position: relative;
  text-align: center;
  min-height: 420px;
}

/* Upper item */

.timeline-top {
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
}

/* Lower item */

.timeline-bottom {
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
}

/* Card */

.timeline-card {
  width: 190px;
  background: #f8fbff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  padding: 1rem;
  transition: all 0.2s ease;
}

.timeline-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 22px rgba(0,0,0,0.05);
  border-color: #c7d7ea;
}

/* Labels */

.timeline-phase {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #185FA5;
  margin-bottom: 0.5rem;
}

.timeline-card-title {
  font-size: 17px;
  font-weight: 600;
  color: #111827;
  margin-bottom: 0.7rem;
  line-height: 1.4;
}

.timeline-card-text {
  font-size: 13px;
  color: #6b7280;
  line-height: 1.7;
}

/* Connector */

.timeline-connector-top {
  width: 2px;
  height: 70px;
  background: #dbe7f3;
  margin: 0 auto;
}

.timeline-connector-bottom {
  width: 2px;
  height: 70px;
  background: #dbe7f3;
  margin: 0 auto;
}

/* Dot */

.timeline-dot {
  width: 16px;
  height: 16px;
  border-radius: 999px;
  background: #185FA5;
  border: 4px solid white;
  box-shadow: 0 0 0 2px #dbe7f3;
  margin: 0 auto;
  position: relative;
  z-index: 3;
}

/* Year */

.timeline-year {
  font-size: 24px;
  font-weight: 700;
  color: #185FA5;
  letter-spacing: -0.03em;
  margin: 0.8rem 0;
}

/* =========================
   RELEASES
========================= */

.release-section {
  border: 1px solid #e5e7eb;
  border-radius: 20px;
  padding: 1.8rem 2rem;
  background: white;
}

.release-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.release-card {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 1.2rem;
  background: #f8fbff;
  transition: all 0.2s ease;
  min-height: 220px;
}

.release-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 22px rgba(0,0,0,0.05);
  border-color: #f8fbff;
}

.release-month {
  font-size: 20px;
  font-weight: 600;
  color: #185FA5;
  margin-bottom: 0.3rem;
}

.release-tag {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #E6F1FB;
  color: #185FA5;
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 1rem;
}

.release-content {
  font-size: 14px;
  color: #6b7280;
  line-height: 1.7;
}

.release-content ul {
  padding-left: 1rem;
  margin-top: 0.6rem;
}

.release-content li {
  margin-bottom: 0.45rem;
}

/* =========================
   RESPONSIVE
========================= */

@media (max-width: 1100px) {

  .timeline-grid {
    grid-template-columns: 1fr;
    gap: 3rem;
  }

  .timeline-line {
    display: none;
  }

  .timeline-item {
    min-height: auto;
  }

  .timeline-top,
  .timeline-bottom {
    position: relative;
    top: auto;
    bottom: auto;
    left: auto;
    transform: none;
  }

}
</style>

<!-- =========================
     HEADER
========================= -->

<div style="
  padding:1.8rem 2rem;
  border-radius:20px;
  border:1px solid #e5e7eb;
  background:white;
  margin-bottom:2rem;
">

<h1 style="
  margin:0;
  font-size:38px;
  font-weight:600;
  color:#185FA5;
  letter-spacing:-0.03em;
">
  Development Notes
</h1>

<p style="
  margin-top:0.8rem;
  max-width:850px;
  font-size:14px;
  color:#6b7280;
  line-height:1.7;
">
  This page documents NDSL releases, milestones,
  bug fixes, performance improvements,
  and future roadmap items for both developers and users.
</p>

</div>

<!-- =========================
     ROADMAP
========================= -->

<div class="roadmap-section">

<p class="roadmap-title-main">
  NDSL Roadmap
</p>


<div class="timeline">

  <div class="timeline-line"></div>

  <div class="timeline-grid">

    <!-- ITEM 1 -->

    <div class="timeline-item">

      <div class="timeline-top">

        <div class="timeline-card">

          <div class="timeline-phase">
            Q3 2026
          </div>

          <div class="timeline-card-title">
            Validation
          </div>

          <div class="timeline-card-text">
            Large-scale GCM runs on CPU and GPU architectures.
          </div>

        </div>

        <div class="timeline-connector-top"></div>

        <div class="timeline-dot"></div>

        <div class="timeline-year">
          2026
        </div>

      </div>

    </div>

    <!-- ITEM 2 -->

    <div class="timeline-item">

      <div class="timeline-bottom">

        <div class="timeline-year">
          2027
        </div>

        <div class="timeline-dot"></div>

        <div class="timeline-connector-bottom"></div>

        <div class="timeline-card">

          <div class="timeline-phase">
            Q1 2027
          </div>

          <div class="timeline-card-title">
            Optimization
          </div>

          <div class="timeline-card-text">
            Infrastructure cleanup and optimization work.
          </div>

        </div>

      </div>

    </div>

    <!-- ITEM 3 -->

    <div class="timeline-item">

      <div class="timeline-top">

        <div class="timeline-card">

          <div class="timeline-phase">
            Q3 2027
          </div>

          <div class="timeline-card-title">
            Performance
          </div>

          <div class="timeline-card-text">
            Improve runtime throughput, scalability, and backend execution efficiency.
          </div>

        </div>

        <div class="timeline-connector-top"></div>

        <div class="timeline-dot"></div>

        <div class="timeline-year">
          2027
        </div>

      </div>

    </div>

    <!-- ITEM 4 -->

    <div class="timeline-item">

      <div class="timeline-bottom">

        <div class="timeline-year">
          2028
        </div>

        <div class="timeline-dot"></div>

        <div class="timeline-connector-bottom"></div>

        <div class="timeline-card">

          <div class="timeline-phase">
            Q1 2028
          </div>

          <div class="timeline-card-title">
            GPU Scaling
          </div>

          <div class="timeline-card-text">
            Multi-node GPU optimization and scalability testing.
          </div>

        </div>

      </div>

    </div>

    <!-- ITEM 5 -->

    <div class="timeline-item">

      <div class="timeline-top">

        <div class="timeline-card">

          <div class="timeline-phase">
            Future
          </div>

          <div class="timeline-card-title">
            Growth
          </div>

          <div class="timeline-card-text">
            Documentation, community expansion, and collaboration.
          </div>

        </div>

        <div class="timeline-connector-top"></div>

        <div class="timeline-dot"></div>

        <div class="timeline-year">
          2028+
        </div>

      </div>

    </div>


  </div>

</div>

</div>

<!-- =========================
     RELEASES
========================= -->

<div class="release-section">

<h2 style="
  margin-top:0;
  color:#185FA5;
  font-size:30px;
  margin-bottom:0.5rem;
">
  2026 Releases
</h2>

<p style="
  color:#6b7280;
  font-size:14px;
  margin-bottom:1.2rem;
">
  Monthly development summaries, release notes, and project updates.
</p>

<div class="release-grid">

  <div class="release-card">

    <div class="release-month">
      May 2026
    </div>

    <div class="release-tag">
      NDSL v2026.05.00
    </div>

    <div class="release-content">
      <ul>
        <li>Add release notes</li>
        <li>Add release notes</li>
        <li>Add release notes</li>
        <li>Add release notes</li>
      </ul>
    </div>

  </div>

  <div class="release-card">

    <div class="release-month">
      June 2026
    </div>

    <div class="release-tag">
      In Progress
    </div>

    <div class="release-content">
      <ul>
        <li>Add release notes</li>
        <li>Add release notes</li>
        <li>Add release notes</li>
        <li>Add release notes</li>
      </ul>
    </div>

  </div>

  <div class="release-card">

    <div class="release-month">
      July 2026
    </div>

    <div class="release-tag">
      Planned
    </div>

    <div class="release-content">
      <ul>
        <li>Add release notes</li>
        <li>Add release notes</li>
        <li>Add release notes</li>
        <li>Add release notes</li>
      </ul>
    </div>

  </div>

</div>

</div>
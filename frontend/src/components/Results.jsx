import React from 'react';
import { BadgeCheck, ChartLine, Wallet } from 'lucide-react';

const steps = [
  { Icon: Wallet, title: 'Enter costs and assumptions', body: 'Add your unit cost, other per-unit costs, and the market inputs you have.' },
  { Icon: ChartLine, title: 'Compare candidate prices', body: 'The model evaluates a range of prices against its market inputs.' },
  { Icon: BadgeCheck, title: 'Review price and profit', body: 'See the recommendation with expected profit, a curve, and market flags.' },
];

export default function Results() {
  return <section id="how-it-works" className="how-it-works">
    <header><p className="eyebrow">How it works</p><h2>Three steps to a clearer price.</h2></header>
    <div className="how-steps">{steps.map(({ Icon, title, body }, index) => <article key={title}><span className="step-icon"><Icon size={19} aria-hidden="true" /></span><span className="step-number">0{index + 1}</span><h3>{title}</h3><p>{body}</p></article>)}</div>
  </section>;
}

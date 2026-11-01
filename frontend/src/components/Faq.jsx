import React from 'react';

const questions = [
  ['What is the difference between markup and margin?', 'Markup is (price − cost) ÷ cost. Margin is (price − cost − fixed costs) ÷ price. They use different denominators, so they are not interchangeable.'],
  ['Where do competitor prices come from?', 'If you enter prices, the optimizer uses those as its market reference. If you leave them blank, it generates simulated competitor prices. The result identifies simulated data.'],
  ['How is the price chosen?', 'The optimizer evaluates candidate prices inside your allowed markup range and selects the price with the highest expected profit according to the active model.'],
  ['How accurate is the model?', 'There is no accuracy guarantee or displayed accuracy score. The model is trained on simulated or sample data unless you supply real competitor prices; results are estimates and depend on the inputs.'],
  ['What does quality tier do?', 'Quality tier is saved as metadata only. It does not change price, demand, or margins.'],
];

export default function Faq() {
  return <section className="faq-section" id="faq">
    <header><p className="eyebrow">Common questions</p><h2>About the recommendation</h2></header>
    <div className="faq-list">{questions.map(([question, answer]) => <details key={question}><summary>{question}</summary><p>{answer}</p></details>)}</div>
  </section>;
}

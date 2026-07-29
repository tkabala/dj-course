import { Component } from '@angular/core';
import { Heading1Component, SubtitleComponent } from '../../../ui-library/Typography/Typography.component';

@Component({
  selector: 'app-reports-header',
  standalone: true,
  imports: [Heading1Component, SubtitleComponent],
  template: `
    <div>
      <ui-heading1>Reports & Analytics</ui-heading1>
      <ui-subtitle>View warehouse performance reports and analytics</ui-subtitle>
    </div>
  `
})
export class ReportsHeaderComponent {}

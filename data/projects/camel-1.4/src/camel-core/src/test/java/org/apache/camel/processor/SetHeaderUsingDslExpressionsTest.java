/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.processor;

import org.apache.camel.ContextTestSupport;
import org.apache.camel.Exchange;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.mock.MockEndpoint;

/**
 * @version $Revision: 647433 $
 */
public class SetHeaderUsingDslExpressionsTest extends ContextTestSupport {
    protected String body = "<person name='James' city='London'/>";
    protected MockEndpoint expected;

    public void testUseConstant() throws Exception {
        context.addRoutes(new RouteBuilder() {
            public void configure() throws Exception {
                from("direct:start").
                        setHeader("foo").constant("ABC").
                        to("mock:result");
            }
        });

        template.sendBodyAndHeader("direct:start", body, "bar", "ABC");

        assertMockEndpointsSatisifed();
    }

    public void testUseConstantParameter() throws Exception {
        context.addRoutes(new RouteBuilder() {
            public void configure() throws Exception {
                from("direct:start").
                        setHeader("foo", "ABC").
                        to("mock:result");
            }
        });

        template.sendBodyAndHeader("direct:start", body, "bar", "ABC");

        assertMockEndpointsSatisifed();
    }

    public void testUseExpression() throws Exception {
        context.addRoutes(new RouteBuilder() {
            public void configure() throws Exception {
                from("direct:start").setHeader("foo").expression(new ExpressionAdapter() {
                    public Object evaluate(Exchange exchange) {
                        return "ABC";
                    }
                }).to("mock:result");
            }
        });

        template.sendBodyAndHeader("direct:start", body, "bar", "ABC");

        assertMockEndpointsSatisifed();
    }

    public void testUseHeaderExpression() throws Exception {
        context.addRoutes(new RouteBuilder() {
            public void configure() throws Exception {
                from("direct:start").
                        setHeader("foo").header("bar").
                        to("mock:result");
            }
        });

        template.sendBodyAndHeader("direct:start", body, "bar", "ABC");

        assertMockEndpointsSatisifed();
    }

    public void testUseBodyExpression() throws Exception {
        context.addRoutes(new RouteBuilder() {
            public void configure() throws Exception {
                from("direct:start").
                        setHeader("foo").body().
                        to("mock:result");
            }
        });

        template.sendBody("direct:start", "ABC");

        assertMockEndpointsSatisifed();
    }

    public void testUseBodyAsTypeExpression() throws Exception {
        context.addRoutes(new RouteBuilder() {
            public void configure() throws Exception {
                from("direct:start").
                        setHeader("foo").body(String.class).
                        to("mock:result");
            }
        });

        template.sendBody("direct:start", "ABC".getBytes());

        assertMockEndpointsSatisifed();
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        expected = getMockEndpoint("mock:result");
        expected.message(0).header("foo").isEqualTo("ABC");
    }
}
